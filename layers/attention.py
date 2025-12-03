"""Attention layer with paged KV-cache for JAX.

Implements multi-head attention with:
- Paged KV-cache for efficient memory management
- Support for both prefill (variable-length) and decode (single-token) phases
- Optimized batched attention using JAX's vectorization
- Flash Attention via jax.nn.dot_product_attention (XLA fused implementation)
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


@partial(jax.jit, donate_argnums=(2, 3))
def store_kv_to_cache(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Store key and value tensors to paged cache (JIT compiled with buffer donation).
    
    Args:
        key: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        value: Value tensor of shape [num_tokens, num_kv_heads, head_dim].
        k_cache: Key cache of shape [num_blocks * block_size, num_kv_heads, head_dim].
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
    
    # Cast to cache dtype for efficient memory bandwidth
    key = key.astype(k_cache.dtype)
    value = value.astype(v_cache.dtype)
    
    # Optimized scatter using JAX's indexed update
    k_cache = k_cache.at[safe_slots].set(key, mode='drop')
    v_cache = v_cache.at[safe_slots].set(value, mode='drop')
    
    return k_cache, v_cache


def gather_kv_from_cache(
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather K/V from paged cache for decode attention (optimized).
    
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
    
    # Gather blocks for each sequence - use take for better performance
    flat_indices = safe_block_tables.reshape(-1)
    gathered_k = jnp.take(k_cache, flat_indices, axis=0)
    gathered_v = jnp.take(v_cache, flat_indices, axis=0)
    
    # Reshape to [batch, max_context_len, heads, dim]
    gathered_k = gathered_k.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    gathered_v = gathered_v.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    
    # Create attention mask based on context_lens
    positions = jnp.arange(max_context_len)[None, :]
    mask = positions < context_lens[:, None]
    
    return gathered_k, gathered_v, mask


# =============================================================================
# Optimized Batched Attention (replaces slow fori_loop)
# =============================================================================

def _make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    """Create a causal attention mask."""
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10))
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
    batch_size: int,
) -> jax.Array:
    """Variable-length attention for prefill - FULLY VECTORIZED.
    
    Uses batched attention with padding and masking for GPU efficiency.
    This replaces the slow fori_loop implementation.
    
    Args:
        q, k, v: Packed tensors [total_tokens, heads, head_dim]
        cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths [batch+1]
        max_seqlen_q, max_seqlen_k: Maximum sequence lengths (for static shapes)
        num_heads, num_kv_heads: Attention head counts
        scale: Softmax scale factor
        batch_size: Number of sequences in batch
    
    Returns:
        Output tensor [total_tokens, num_heads, head_dim]
    """
    head_dim = q.shape[-1]
    total_tokens = q.shape[0]
    
    # Note: GQA is handled by jax.nn.dot_product_attention via broadcasting
    # No need to expand KV heads - it supports num_heads != num_kv_heads natively
    
    # For single sequence, use jax.nn.dot_product_attention directly
    if batch_size == 1:
        # q: [seq_len, num_heads, head_dim] -> [1, seq_len, num_heads, head_dim]
        # k, v: [seq_len, num_kv_heads, head_dim] -> [1, seq_len, num_kv_heads, head_dim]
        seq_len = total_tokens
        q_batched = q[None, :, :, :]  # [1, seq, heads, dim]
        k_batched = k[None, :, :, :]  # [1, seq, kv_heads, dim]
        v_batched = v[None, :, :, :]  # [1, seq, kv_heads, dim]
        
        # Causal mask: [1, 1, seq, seq]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        mask = causal_mask[None, None, :, :]  # [1, 1, seq, seq]
        
        # Use fused attention - handles GQA via broadcasting
        output = jax.nn.dot_product_attention(
            q_batched, k_batched, v_batched,
            mask=mask,
            scale=scale,
        )  # [1, seq, num_heads, dim]
        
        return output.squeeze(0)  # [seq, num_heads, dim]
    
    # ==========================================================================
    # VECTORIZED padding using advanced indexing (replaces O(N) fori_loop)
    # ==========================================================================
    
    # Compute sequence lengths
    seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]  # [batch]
    seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]  # [batch]
    
    # Create index arrays for vectorized gather
    # For each (batch, position), compute the source index in packed array
    pos_indices_q = jnp.arange(max_seqlen_q)[None, :]  # [1, max_seq_q]
    pos_indices_k = jnp.arange(max_seqlen_k)[None, :]  # [1, max_seq_k]
    
    # Source indices: cu_seqlens[i] + pos for each sequence
    # Shape: [batch, max_seq]
    q_src_indices = cu_seqlens_q[:-1, None] + pos_indices_q  # [batch, max_seq_q]
    k_src_indices = cu_seqlens_k[:-1, None] + pos_indices_k  # [batch, max_seq_k]
    
    # Clamp indices to valid range (out-of-bounds will be masked anyway)
    q_src_indices = jnp.clip(q_src_indices, 0, total_tokens - 1)
    k_src_indices = jnp.clip(k_src_indices, 0, total_tokens - 1)
    
    # Validity masks for padding
    q_valid = pos_indices_q < seq_lens_q[:, None]  # [batch, max_seq_q]
    k_valid = pos_indices_k < seq_lens_k[:, None]  # [batch, max_seq_k]
    
    # Vectorized gather using advanced indexing - O(1) parallel operation!
    # q: [total_tokens, heads, dim] -> q_padded: [batch, max_seq, heads, dim]
    q_padded = q[q_src_indices]  # [batch, max_seq_q, heads, dim]
    k_padded = k[k_src_indices]  # [batch, max_seq_k, heads, dim]
    v_padded = v[k_src_indices]  # [batch, max_seq_k, heads, dim]
    
    # Zero out invalid (padding) positions
    q_padded = jnp.where(q_valid[:, :, None, None], q_padded, 0)
    k_padded = jnp.where(k_valid[:, :, None, None], k_padded, 0)
    v_padded = jnp.where(k_valid[:, :, None, None], v_padded, 0)
    
    # Create attention mask combining causal + padding
    # [batch, max_seq_q, max_seq_k]
    q_pos = jnp.arange(max_seqlen_q)[None, :, None]  # [1, seq_q, 1]
    k_pos = jnp.arange(max_seqlen_k)[None, None, :]  # [1, 1, seq_k]
    
    # Causal mask: q_pos >= k_pos
    causal_mask = q_pos >= k_pos  # [1, seq_q, seq_k]
    
    # Padding mask: only attend to valid positions
    q_valid = jnp.arange(max_seqlen_q)[None, :] < seq_lens_q[:, None]  # [batch, seq_q]
    k_valid = jnp.arange(max_seqlen_k)[None, :] < seq_lens_k[:, None]  # [batch, seq_k]
    padding_mask = q_valid[:, :, None] & k_valid[:, None, :]  # [batch, seq_q, seq_k]
    
    # Combined mask: [batch, seq_q, seq_k]
    full_mask = causal_mask & padding_mask
    
    # Use jax.nn.dot_product_attention for fused implementation
    # Input shape for BNTS format: [batch, seq, heads, dim]
    # Mask shape: [batch, 1, seq_q, seq_k] for broadcasting over heads
    mask_for_attn = full_mask[:, None, :, :]  # Add head dimension
    
    output_padded = jax.nn.dot_product_attention(
        q_padded, k_padded, v_padded,
        mask=mask_for_attn,
        scale=scale,
    )  # [batch, max_seq_q, heads, dim]
    
    # ==========================================================================
    # VECTORIZED scatter back to packed output (replaces O(N) fori_loop)
    # ==========================================================================
    
    # Compute destination indices for each (batch, position) -> packed index
    # dest_indices[b, p] = cu_seqlens_q[b] + p
    dest_indices = cu_seqlens_q[:-1, None] + pos_indices_q  # [batch, max_seq_q]
    dest_indices = jnp.clip(dest_indices, 0, total_tokens - 1)
    
    # Flatten for scatter operation
    dest_flat = dest_indices.reshape(-1)  # [batch * max_seq_q]
    output_flat = output_padded.reshape(-1, num_heads, head_dim)  # [batch * max_seq_q, heads, dim]
    valid_flat = q_valid.reshape(-1)  # [batch * max_seq_q]
    
    # Initialize output and scatter valid entries using vectorized operation
    output = jnp.zeros((total_tokens, num_heads, head_dim), dtype=q.dtype)
    
    # Masked scatter: only write valid positions
    # Use .at[].set() with masked values - invalid positions write zeros which we overwrite
    masked_output = jnp.where(valid_flat[:, None, None], output_flat, 0)
    output = output.at[dest_flat].add(masked_output)
    
    return output


class Attention(nnx.Module):
    """Multi-head attention with paged KV-cache.
    
    Supports both prefill (processing full sequences) and decode (single token)
    phases with paged attention for efficient memory usage.
    
    Optimized for GPU efficiency using:
    - jax.nn.dot_product_attention (XLA fused Flash Attention)
    - Batched operations instead of sequential loops
    - Efficient KV-cache scatter operations
    
    Attributes:
        num_heads: Number of query attention heads.
        head_dim: Dimension of each attention head.
        scale: Softmax scale factor (typically 1/sqrt(head_dim)).
        num_kv_heads: Number of key/value heads (for GQA/MQA).
        k_cache: Reference to layer's key cache (set by model runner).
        v_cache: Reference to layer's value cache (set by model runner).
        block_size: Number of tokens per cache block.
    """
    
    # Declare cache attributes as data (mutable) for Flax NNX 0.12+
    k_cache: jax.Array | None = nnx.data()
    v_cache: jax.Array | None = nnx.data()
    
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
        self.k_cache = None
        self.v_cache = None
    
    def set_kv_cache(self, k_cache: jax.Array, v_cache: jax.Array):
        """Set KV cache references (called by model runner)."""
        self.k_cache = k_cache
        self.v_cache = v_cache
    
    def _prefill_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Attention for prefill phase with variable-length sequences.
        
        Uses optimized batched attention with jax.nn.dot_product_attention.
        
        Args:
            q: Query tensor [total_tokens, num_heads, head_dim].
            k: Key tensor [total_tokens, num_kv_heads, head_dim].
            v: Value tensor [total_tokens, num_kv_heads, head_dim].
            context: Attention context with sequence boundaries.
        
        Returns:
            Output tensor [total_tokens, num_heads, head_dim].
        """
        batch_size = context.cu_seqlens_q.shape[0] - 1
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
            batch_size=batch_size,
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
            self.k_cache,
            self.v_cache,
            context.block_tables,
            context.context_lens,
            self.block_size,
        )
        
        # q: [batch, heads, dim] -> [batch, 1, heads, dim] for dot_product_attention
        # Cast q/k/v to the cache dtype (bfloat16) for consistent ops
        target_dtype = self.k_cache.dtype if self.k_cache is not None else q.dtype
        q = q.astype(target_dtype)
        k_gathered = k_gathered.astype(target_dtype)
        v_gathered = v_gathered.astype(target_dtype)
        q = q[:, None, :, :]
        
        # Note: GQA is handled by jax.nn.dot_product_attention via broadcasting
        # Q: [batch, 1, num_heads, dim], K/V: [batch, seq, num_kv_heads, dim]
        # No need to expand KV heads
        
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
        # Store K/V to cache (use views to avoid copies)
        if self.k_cache is not None and self.v_cache is not None:
            num_blocks = self.k_cache.shape[0]
            cache_shape = (num_blocks * self.block_size, self.num_kv_heads, self.head_dim)
            
            # Use reshape (no-copy view) for slot-based indexing
            k_cache_flat = self.k_cache.reshape(cache_shape)
            v_cache_flat = self.v_cache.reshape(cache_shape)
            
            k_cache_flat, v_cache_flat = store_kv_to_cache(
                k, v, k_cache_flat, v_cache_flat, context.slot_mapping
            )
            
            # Reshape back (no-copy view)
            self.k_cache = k_cache_flat.reshape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim
            )
            self.v_cache = v_cache_flat.reshape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim
            )
        
        if context.is_prefill:
            return self._prefill_attention(q, k, v, context)
        else:
            return self._decode_attention(q, context)
        
        if context.is_prefill:
            return self._prefill_attention(q, k, v, context)
        else:
            return self._decode_attention(q, context)
