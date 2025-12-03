"""Attention layer with paged KV-cache for JAX.

Implements multi-head attention with:
- Paged KV-cache for efficient memory management
- Support for both prefill (variable-length) and decode (single-token) phases
- Uses jax.nn.dot_product_attention for computation
- Variable-length sequence handling via jax.lax.dynamic_slice (memory-efficient)
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
from typing import NamedTuple
from functools import partial

from nanovllm_jax.utils.context import AttentionContext


class KVCache(NamedTuple):
    """KV cache storage for a single layer.
    
    Attributes:
        k_cache: Key cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Value cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
    """
    k_cache: jax.Array
    v_cache: jax.Array


def store_kv_to_cache(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Store key and value tensors to paged cache.
    
    This is the pure JAX equivalent of the Triton kernel. It scatters
    K/V vectors to non-contiguous cache slots.
    
    Args:
        key: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        value: Value tensor of shape [num_tokens, num_kv_heads, head_dim].
        k_cache: Key cache of shape [num_blocks * block_size, num_kv_heads, head_dim]
                 (flattened view for easier indexing).
        v_cache: Value cache with same shape as k_cache.
        slot_mapping: Mapping from token index to cache slot [num_tokens].
                     -1 indicates skip (already cached).
    
    Returns:
        Updated (k_cache, v_cache) tuple.
    """
    # Create mask for valid slots (not -1)
    valid_mask = slot_mapping >= 0
    
    # Replace -1 with 0 for indexing (will be masked out anyway)
    safe_slots = jnp.where(valid_mask, slot_mapping, 0)
    
    # Efficient vectorized scatter using .at[].set()
    k_cache = k_cache.at[safe_slots].set(
        jnp.where(valid_mask[:, None, None], key, k_cache[safe_slots])
    )
    v_cache = v_cache.at[safe_slots].set(
        jnp.where(valid_mask[:, None, None], value, v_cache[safe_slots])
    )
    
    return k_cache, v_cache


def gather_kv_from_cache(
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather K/V from paged cache for decode attention.
    
    Args:
        k_cache: Key cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Value cache with same shape.
        block_tables: Block indices for each sequence [batch_size, max_blocks].
        context_lens: Length of context for each sequence [batch_size].
        block_size: Number of tokens per block.
    
    Returns:
        Tuple of (keys, values, mask) where keys/values have shape 
        [batch_size, max_context_len, num_kv_heads, head_dim].
    """
    batch_size = block_tables.shape[0]
    max_blocks = block_tables.shape[1]
    max_context_len = max_blocks * block_size
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    
    # Clamp block indices to valid range for gathering
    safe_block_tables = jnp.clip(block_tables, 0, k_cache.shape[0] - 1)
    
    # Gather blocks for each sequence
    gathered_k = k_cache[safe_block_tables.reshape(-1)]
    gathered_v = v_cache[safe_block_tables.reshape(-1)]
    
    # Reshape to [batch, max_context_len, heads, dim]
    gathered_k = gathered_k.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    gathered_v = gathered_v.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    
    # Create attention mask based on context_lens
    positions = jnp.arange(max_context_len)[None, :]
    mask = positions < context_lens[:, None]
    
    return gathered_k, gathered_v, mask


# =============================================================================
# Variable-Length Sequence Attention Using dynamic_slice
# =============================================================================

def _single_sequence_attention(
    q: jax.Array,  # [seq_len_q, num_heads, head_dim]
    k: jax.Array,  # [seq_len_k, num_kv_heads, head_dim]
    v: jax.Array,  # [seq_len_k, num_kv_heads, head_dim]
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> jax.Array:
    """Compute attention for a single sequence using dynamic shapes.
    
    Uses jax.lax.dynamic_slice for memory-efficient variable-length handling.
    """
    head_dim = q.shape[-1]
    
    # Transpose to [heads, seq, dim]
    q = jnp.transpose(q, (1, 0, 2))  # [num_heads, seq_q, head_dim]
    k = jnp.transpose(k, (1, 0, 2))  # [num_kv_heads, seq_k, head_dim]
    v = jnp.transpose(v, (1, 0, 2))  # [num_kv_heads, seq_k, head_dim]
    
    # Handle GQA: repeat KV heads
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        k = jnp.repeat(k, num_groups, axis=0)
        v = jnp.repeat(v, num_groups, axis=0)
    
    # Compute attention scores: [num_heads, seq_q, seq_k]
    scores = jnp.einsum('hqd,hkd->hqk', q, k) * scale
    
    # Create causal mask using dynamic_slice-friendly indexing
    # Build mask where position i can attend to positions 0..i
    q_pos = jnp.arange(seq_len_q)[:, None]
    k_pos = jnp.arange(seq_len_k)[None, :]
    causal_mask = q_pos >= k_pos  # [seq_q, seq_k]
    
    # Apply causal mask
    scores = jnp.where(causal_mask[None, :, :], scores, -1e9)
    
    # Softmax and attention output
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v)  # [num_heads, seq_q, head_dim]
    
    # Transpose back to [seq_q, num_heads, head_dim]
    return jnp.transpose(output, (1, 0, 2))


def variable_length_attention_prefill(
    q: jax.Array,  # [total_tokens, num_heads, head_dim]
    k: jax.Array,  # [total_tokens, num_kv_heads, head_dim]
    v: jax.Array,  # [total_tokens, num_kv_heads, head_dim]
    cu_seqlens_q: jax.Array,  # [batch_size + 1]
    cu_seqlens_k: jax.Array,  # [batch_size + 1]
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> jax.Array:
    """Variable-length attention for prefill.
    
    Uses a simple batched approach with padding and masking.
    Optimized for memory efficiency by avoiding large intermediate allocations.
    
    Args:
        q, k, v: Packed tensors [total_tokens, heads, head_dim]
        cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths [batch+1]
        max_seqlen_q, max_seqlen_k: Maximum sequence lengths (for static shapes)
        num_heads, num_kv_heads: Attention head counts
        scale: Softmax scale factor
    
    Returns:
        Output tensor [total_tokens, num_heads, head_dim]
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    head_dim = q.shape[-1]
    total_tokens = q.shape[0]
    
    # Process each sequence using lax.fori_loop for memory efficiency
    # This avoids creating large padded batched tensors
    
    def process_single_seq(i, output):
        """Process a single sequence and update output in-place."""
        start_q = cu_seqlens_q[i]
        end_q = cu_seqlens_q[i + 1]
        len_q = end_q - start_q
        
        start_k = cu_seqlens_k[i]
        end_k = cu_seqlens_k[i + 1]
        len_k = end_k - start_k
        
        # Extract Q, K, V for this sequence using dynamic_slice
        # We extract max_seqlen tokens but only use len tokens
        q_seq = lax.dynamic_slice(q, (start_q, 0, 0), (max_seqlen_q, num_heads, head_dim))
        k_seq = lax.dynamic_slice(k, (start_k, 0, 0), (max_seqlen_k, num_kv_heads, head_dim))
        v_seq = lax.dynamic_slice(v, (start_k, 0, 0), (max_seqlen_k, num_kv_heads, head_dim))
        
        # Handle GQA: repeat KV heads
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            k_seq = jnp.repeat(k_seq, num_groups, axis=1)
            v_seq = jnp.repeat(v_seq, num_groups, axis=1)
        
        # Transpose to [heads, seq, dim] for attention
        q_seq = jnp.transpose(q_seq, (1, 0, 2))  # [num_heads, max_seq_q, head_dim]
        k_seq = jnp.transpose(k_seq, (1, 0, 2))  # [num_heads, max_seq_k, head_dim]
        v_seq = jnp.transpose(v_seq, (1, 0, 2))  # [num_heads, max_seq_k, head_dim]
        
        # Compute attention scores: [num_heads, max_seq_q, max_seq_k]
        scores = jnp.einsum('hqd,hkd->hqk', q_seq, k_seq) * scale
        
        # Create masks
        # 1. Causal mask: position q can attend to k where k <= q
        q_pos = jnp.arange(max_seqlen_q)[:, None]
        k_pos = jnp.arange(max_seqlen_k)[None, :]
        causal_mask = q_pos >= k_pos  # [max_seq_q, max_seq_k]
        
        # 2. Padding mask: only attend to valid positions
        q_valid = jnp.arange(max_seqlen_q) < len_q  # [max_seq_q]
        k_valid = jnp.arange(max_seqlen_k) < len_k  # [max_seq_k]
        padding_mask = q_valid[:, None] & k_valid[None, :]  # [max_seq_q, max_seq_k]
        
        # Combined mask
        full_mask = causal_mask & padding_mask
        
        # Apply mask with large negative value for softmax
        scores = jnp.where(full_mask[None, :, :], scores, -1e9)
        
        # Softmax and weighted sum
        attn_weights = jax.nn.softmax(scores, axis=-1)
        seq_output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_seq)  # [num_heads, max_seq_q, head_dim]
        
        # Transpose back: [max_seq_q, num_heads, head_dim]
        seq_output = jnp.transpose(seq_output, (1, 0, 2))
        
        # Mask output for valid positions only
        valid_mask = jnp.arange(max_seqlen_q) < len_q
        seq_output = jnp.where(valid_mask[:, None, None], seq_output, 0.0)
        
        # Scatter to output using segment-based approach
        positions = start_q + jnp.arange(max_seqlen_q)
        output = output.at[positions].add(seq_output)
        
        return output
    
    output = jnp.zeros((total_tokens, num_heads, head_dim), dtype=q.dtype)
    output = lax.fori_loop(0, batch_size, process_single_seq, output)
    
    return output


class Attention(nnx.Module):
    """Multi-head attention with paged KV-cache.
    
    Supports both prefill (processing full sequences) and decode (single token)
    phases with paged attention for efficient memory usage.
    
    Uses jax.lax.dynamic_slice for memory-efficient variable-length handling
    in prefill phase without wasteful padding.
    
    Attributes:
        num_heads: Number of query attention heads.
        head_dim: Dimension of each attention head.
        scale: Softmax scale factor (typically 1/sqrt(head_dim)).
        num_kv_heads: Number of key/value heads (for GQA/MQA).
        k_cache: Reference to layer's key cache (set by model runner).
        v_cache: Reference to layer's value cache (set by model runner).
        block_size: Number of tokens per cache block.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        block_size: int = 256,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        
        # KV cache will be set by model runner after allocation
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        self.k_cache: nnx.Variable | None = None
        self.v_cache: nnx.Variable | None = None
    
    def set_kv_cache(self, k_cache: jax.Array, v_cache: jax.Array):
        """Set KV cache references (called by model runner)."""
        self.k_cache = nnx.Variable(k_cache)
        self.v_cache = nnx.Variable(v_cache)
    
    def _prefill_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Attention for prefill phase with variable-length sequences.
        
        Uses jax.lax.dynamic_slice for memory-efficient variable-length handling.
        
        Args:
            q: Query tensor [total_tokens, num_heads, head_dim].
            k: Key tensor [total_tokens, num_kv_heads, head_dim].
            v: Value tensor [total_tokens, num_kv_heads, head_dim].
            context: Attention context with sequence boundaries.
        
        Returns:
            Output tensor [total_tokens, num_heads, head_dim].
        """
        return variable_length_attention_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
        )
    
    def _decode_attention(
        self,
        q: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Attention for decode phase with single query token per sequence.
        
        Args:
            q: Query tensor [batch_size, num_heads, head_dim].
            context: Attention context with cache info.
        
        Returns:
            Output tensor [batch_size, num_heads, head_dim].
        """
        # Gather K/V from paged cache
        # k_gathered, v_gathered: [batch, max_context_len, kv_heads, dim]
        # kv_mask: [batch, max_context_len]
        k_gathered, v_gathered, kv_mask = gather_kv_from_cache(
            self.k_cache.value,
            self.v_cache.value,
            context.block_tables,
            context.context_lens,
            self.block_size,
        )
        
        # q: [batch, heads, dim] -> [batch, 1, heads, dim] for dot_product_attention
        q = q[:, None, :, :]
        
        # Handle GQA: repeat KV heads to match query heads
        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k_gathered = jnp.repeat(k_gathered, num_groups, axis=2)  # repeat along heads dim
            v_gathered = jnp.repeat(v_gathered, num_groups, axis=2)
        
        # Mask: [batch, max_len] -> [batch, 1, 1, max_len] for broadcasting
        # jax.nn.dot_product_attention expects [B, N, T, S] or [B, 1, T, S] mask
        mask = kv_mask[:, None, None, :]  # [batch, 1, 1, max_len]
        
        # Attention with inputs in [batch, seq, heads, dim] format
        output = jax.nn.dot_product_attention(
            q, k_gathered, v_gathered,
            mask=mask,
            scale=self.scale,
        )  # [batch, 1, heads, dim]
        
        # [batch, 1, heads, dim] -> [batch, heads, dim]
        return output.squeeze(1)
    
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Forward pass for attention.
        
        Handles both prefill and decode phases based on context.
        
        Args:
            q: Query tensor [num_tokens, num_heads, head_dim].
            k: Key tensor [num_tokens, num_kv_heads, head_dim].
            v: Value tensor [num_tokens, num_kv_heads, head_dim].
            context: Attention context with phase info and metadata.
        
        Returns:
            Output tensor with same shape as query.
        """
        # Store K/V to cache
        if self.k_cache is not None and self.v_cache is not None:
            # Flatten cache for slot-based indexing
            k_cache_flat = self.k_cache.value.reshape(-1, self.num_kv_heads, self.head_dim)
            v_cache_flat = self.v_cache.value.reshape(-1, self.num_kv_heads, self.head_dim)
            
            k_cache_flat, v_cache_flat = store_kv_to_cache(
                k, v, k_cache_flat, v_cache_flat, context.slot_mapping
            )
            
            # Reshape back and update
            num_blocks = self.k_cache.value.shape[0]
            self.k_cache.value = k_cache_flat.reshape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim
            )
            self.v_cache.value = v_cache_flat.reshape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim
            )
        
        if context.is_prefill:
            return self._prefill_attention(q, k, v, context)
        else:
            return self._decode_attention(q, context)
