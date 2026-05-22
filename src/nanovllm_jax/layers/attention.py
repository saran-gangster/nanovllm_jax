"""Attention layer with paged KV-cache for JAX.

Implements multi-head attention with:
- Paged KV-cache for efficient memory management
- Support for both prefill (variable-length) and decode (single-token) phases
- Optimized batched attention using JAX's vectorization
- Flash Attention via jax.nn.dot_product_attention (XLA fused implementation)
- Pallas kernels for high-performance paged decode attention (when available)
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import core as jax_core
from flax import nnx
from typing import NamedTuple
from functools import partial
from time import perf_counter

from nanovllm_jax.utils.context import AttentionContext
from nanovllm_jax.utils.runtime_diagnostics import (
    block_until_ready_tree,
    decode_step_profiling_enabled,
    record_kv_update_stats,
)

# Try to import paged attention kernels
try:
    from nanovllm_jax.layers.paged_attention import (
        paged_attention as pallas_paged_attention,
        PALLAS_AVAILABLE,
    )
except ImportError:
    PALLAS_AVAILABLE = False
    pallas_paged_attention = None

# Prefill attention backend: auto (cudnn on GPU for bf16/fp16), xla, or cudnn
_PREFILL_ATTN_IMPL_ENV = os.environ.get("NANOVLLM_JAX_PREFILL_ATTN_IMPL", "auto").strip().lower()
if _PREFILL_ATTN_IMPL_ENV == "auto":
    _PREFILL_ATTN_IMPL = "cudnn" if jax.default_backend() == "gpu" else None
elif _PREFILL_ATTN_IMPL_ENV in {"xla", "cudnn"}:
    _PREFILL_ATTN_IMPL = _PREFILL_ATTN_IMPL_ENV
else:
    _PREFILL_ATTN_IMPL = None

_KV_UPDATE_BACKEND = os.environ.get(
    "NANOVLLM_JAX_KV_UPDATE_BACKEND", "scatter"
).strip().lower()


def configure_prefill_backend(config) -> None:
    """Apply Config-driven prefill backend selection.

    Call once at engine startup.  ``"auto"`` keeps env-var default.
    """
    global _PREFILL_ATTN_IMPL
    backend = getattr(config, "prefill_attention_backend", "auto")
    if backend == "auto":
        return
    if backend in {"xla", "cudnn"}:
        _PREFILL_ATTN_IMPL = backend
    elif backend == "mosaic":
        _PREFILL_ATTN_IMPL = None  # Mosaic prefill handled separately
    else:
        _PREFILL_ATTN_IMPL = None


def _prefill_attn_impl_for_dtype(dtype: jnp.dtype) -> str | None:
    """Pick a safe attention backend for the given tensor dtype."""
    if _PREFILL_ATTN_IMPL == "cudnn" and dtype not in (jnp.float16, jnp.bfloat16):
        return None  # cuDNN only supports bf16/fp16
    return _PREFILL_ATTN_IMPL


def _normalize_kv_update_backend(raw: str) -> str:
    backend = str(raw).strip().lower()
    if backend in {"auto", "scatter", "compact_scatter", "sorted_compact_scatter"}:
        return backend
    raise ValueError(
        "NANOVLLM_JAX_KV_UPDATE_BACKEND must be one of: "
        "auto|scatter|compact_scatter|sorted_compact_scatter"
    )


def _active_kv_update_backend() -> str:
    backend = _normalize_kv_update_backend(
        os.environ.get("NANOVLLM_JAX_KV_UPDATE_BACKEND", _KV_UPDATE_BACKEND)
    )
    if backend == "auto":
        return "scatter"
    return backend


class KVCache(NamedTuple):
    """KV cache storage for a single layer.

    Attributes:
        k_cache: Key cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Value cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
    """
    k_cache: jax.Array
    v_cache: jax.Array


# ---------------------------------------------------------------------------
# INT8 KV cache quantization helpers
# ---------------------------------------------------------------------------

_QMAX = 127.0
_QUANT_EPS = 1e-8


def quantize_kv_int8(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Symmetric INT8 quantization per token per KV head.

    Args:
        x: [..., num_kv_heads, head_dim] float tensor.

    Returns:
        (q, scale):
            q: same shape as x, dtype int8, values in [-127, 127].
            scale: [..., num_kv_heads], dtype float16.
    """
    x_f32 = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x_f32), axis=-1)  # [..., num_kv_heads]
    scale = jnp.maximum(amax / _QMAX, _QUANT_EPS)
    q = jnp.rint(x_f32 / scale[..., None])
    q = jnp.clip(q, min=-_QMAX, max=_QMAX).astype(jnp.int8)
    return q, scale.astype(jnp.float16)


def dequantize_kv_int8(
    q: jax.Array,
    scale: jax.Array,
    out_dtype=jnp.bfloat16,
) -> jax.Array:
    """Dequantize INT8 KV cache.

    Args:
        q: [..., num_kv_heads, head_dim], int8.
        scale: [..., num_kv_heads], fp16/bf16.
        out_dtype: Output dtype.

    Returns:
        Dequantized tensor in out_dtype, same shape as q.
    """
    return (q.astype(jnp.float32) * scale.astype(jnp.float32)[..., None]).astype(out_dtype)


# ---------------------------------------------------------------------------
# KV cache store functions
# ---------------------------------------------------------------------------

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
    valid_mask = slot_mapping >= 0
    safe_slots = jnp.where(valid_mask, slot_mapping, k_cache.shape[0])

    if key.dtype != k_cache.dtype:
        key = key.astype(k_cache.dtype)
    if value.dtype != v_cache.dtype:
        value = value.astype(v_cache.dtype)

    k_cache = k_cache.at[safe_slots].set(key, mode='drop')
    v_cache = v_cache.at[safe_slots].set(value, mode='drop')

    return k_cache, v_cache


@partial(jax.jit, donate_argnums=(2, 3))
def store_kv_to_cache_compact_scatter(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Store K/V after compacting valid rows ahead of scatter.

    This keeps invalid rows out of the hot portion of the scatter input while
    preserving a static shape suitable for JIT. It remains internal and
    experimental until GPU validation exists.
    """
    valid_mask = slot_mapping >= 0
    num_tokens = slot_mapping.shape[0]
    valid_positions = jnp.nonzero(valid_mask, size=num_tokens, fill_value=0)[0]
    valid_count = jnp.sum(valid_mask.astype(jnp.int32))
    compact_valid_mask = jnp.arange(num_tokens, dtype=jnp.int32) < valid_count

    key_compact = key[valid_positions]
    value_compact = value[valid_positions]
    slot_compact = slot_mapping[valid_positions]

    if key_compact.dtype != k_cache.dtype:
        key_compact = key_compact.astype(k_cache.dtype)
    if value_compact.dtype != v_cache.dtype:
        value_compact = value_compact.astype(v_cache.dtype)

    key_compact = jnp.where(compact_valid_mask[:, None, None], key_compact, 0)
    value_compact = jnp.where(compact_valid_mask[:, None, None], value_compact, 0)
    safe_slots = jnp.where(compact_valid_mask, slot_compact, k_cache.shape[0])

    k_cache = k_cache.at[safe_slots].set(key_compact, mode="drop")
    v_cache = v_cache.at[safe_slots].set(value_compact, mode="drop")

    return k_cache, v_cache


def _sorted_compact_slots(
    slot_mapping: jax.Array,
    invalid_slot: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_tokens = slot_mapping.shape[0]
    valid_mask = slot_mapping >= 0
    valid_positions = jnp.nonzero(valid_mask, size=num_tokens, fill_value=0)[0]
    valid_count = jnp.sum(valid_mask.astype(jnp.int32))
    compact_valid_mask = jnp.arange(num_tokens, dtype=jnp.int32) < valid_count
    slot_compact = slot_mapping[valid_positions]
    slot_compact = jnp.where(compact_valid_mask, slot_compact, invalid_slot)
    sort_order = jnp.argsort(slot_compact, stable=True)
    sorted_slots = slot_compact[sort_order]
    sorted_valid_mask = sorted_slots < invalid_slot
    return valid_positions, sort_order, sorted_valid_mask


@partial(jax.jit, donate_argnums=(2, 3))
def store_kv_to_cache_sorted_compact_scatter(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Store K/V after compacting valid rows and sorting target slots.

    This keeps invalid rows out of the hot scatter path and presents writes in
    slot order, which is a better approximation of a dedicated cache-update
    operator than plain compaction while staying CPU-safe and JIT-friendly.
    """
    invalid_slot = k_cache.shape[0]
    valid_positions, sort_order, sorted_valid_mask = _sorted_compact_slots(
        slot_mapping,
        invalid_slot,
    )

    key_sorted = key[valid_positions][sort_order]
    value_sorted = value[valid_positions][sort_order]
    slot_sorted = slot_mapping[valid_positions][sort_order]

    if key_sorted.dtype != k_cache.dtype:
        key_sorted = key_sorted.astype(k_cache.dtype)
    if value_sorted.dtype != v_cache.dtype:
        value_sorted = value_sorted.astype(v_cache.dtype)

    key_sorted = jnp.where(sorted_valid_mask[:, None, None], key_sorted, 0)
    value_sorted = jnp.where(sorted_valid_mask[:, None, None], value_sorted, 0)
    safe_slots = jnp.where(sorted_valid_mask, slot_sorted, invalid_slot)

    k_cache = k_cache.at[safe_slots].set(key_sorted, mode="drop")
    v_cache = v_cache.at[safe_slots].set(value_sorted, mode="drop")
    return k_cache, v_cache


@partial(jax.jit, donate_argnums=(2, 3, 4, 5))
def store_kv_to_cache_int8(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    k_scale: jax.Array,
    v_scale: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Store quantized key/value to INT8 paged cache.

    Args:
        key, value: [num_tokens, num_kv_heads, head_dim] float tensors.
        k_cache, v_cache: [num_slots, num_kv_heads, head_dim] int8 cache.
        k_scale, v_scale: [num_slots, num_kv_heads] fp16 scales.
        slot_mapping: [num_tokens], -1 means skip.

    Returns:
        (k_cache, v_cache, k_scale, v_scale) updated.
    """
    valid_mask = slot_mapping >= 0
    safe_slots = jnp.where(valid_mask, slot_mapping, k_cache.shape[0])

    # Zero invalid rows before quantization to avoid NaN propagation.
    key = jnp.where(valid_mask[:, None, None], key, 0)
    value = jnp.where(valid_mask[:, None, None], value, 0)

    k_q, k_s = quantize_kv_int8(key)
    v_q, v_s = quantize_kv_int8(value)

    k_cache = k_cache.at[safe_slots].set(k_q, mode='drop')
    v_cache = v_cache.at[safe_slots].set(v_q, mode='drop')
    k_scale = k_scale.at[safe_slots].set(k_s, mode='drop')
    v_scale = v_scale.at[safe_slots].set(v_s, mode='drop')

    return k_cache, v_cache, k_scale, v_scale


@partial(jax.jit, donate_argnums=(2, 3, 4, 5))
def store_kv_to_cache_int8_compact_scatter(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    k_scale: jax.Array,
    v_scale: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """INT8 KV-cache store after compacting valid rows."""
    valid_mask = slot_mapping >= 0
    num_tokens = slot_mapping.shape[0]
    valid_positions = jnp.nonzero(valid_mask, size=num_tokens, fill_value=0)[0]
    valid_count = jnp.sum(valid_mask.astype(jnp.int32))
    compact_valid_mask = jnp.arange(num_tokens, dtype=jnp.int32) < valid_count

    key_compact = key[valid_positions]
    value_compact = value[valid_positions]
    slot_compact = slot_mapping[valid_positions]

    key_compact = jnp.where(compact_valid_mask[:, None, None], key_compact, 0)
    value_compact = jnp.where(compact_valid_mask[:, None, None], value_compact, 0)

    k_q, k_s = quantize_kv_int8(key_compact)
    v_q, v_s = quantize_kv_int8(value_compact)

    safe_slots = jnp.where(compact_valid_mask, slot_compact, k_cache.shape[0])
    k_cache = k_cache.at[safe_slots].set(k_q, mode="drop")
    v_cache = v_cache.at[safe_slots].set(v_q, mode="drop")
    k_scale = k_scale.at[safe_slots].set(k_s, mode="drop")
    v_scale = v_scale.at[safe_slots].set(v_s, mode="drop")

    return k_cache, v_cache, k_scale, v_scale


@partial(jax.jit, donate_argnums=(2, 3, 4, 5))
def store_kv_to_cache_int8_sorted_compact_scatter(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    k_scale: jax.Array,
    v_scale: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """INT8 KV-cache store after compaction plus slot ordering."""
    invalid_slot = k_cache.shape[0]
    valid_positions, sort_order, sorted_valid_mask = _sorted_compact_slots(
        slot_mapping,
        invalid_slot,
    )

    key_sorted = key[valid_positions][sort_order]
    value_sorted = value[valid_positions][sort_order]
    slot_sorted = slot_mapping[valid_positions][sort_order]

    key_sorted = jnp.where(sorted_valid_mask[:, None, None], key_sorted, 0)
    value_sorted = jnp.where(sorted_valid_mask[:, None, None], value_sorted, 0)

    k_q, k_s = quantize_kv_int8(key_sorted)
    v_q, v_s = quantize_kv_int8(value_sorted)

    safe_slots = jnp.where(sorted_valid_mask, slot_sorted, invalid_slot)
    k_cache = k_cache.at[safe_slots].set(k_q, mode="drop")
    v_cache = v_cache.at[safe_slots].set(v_q, mode="drop")
    k_scale = k_scale.at[safe_slots].set(k_s, mode="drop")
    v_scale = v_scale.at[safe_slots].set(v_s, mode="drop")
    return k_cache, v_cache, k_scale, v_scale


def _tree_has_tracer(value) -> bool:
    return any(
        isinstance(leaf, jax_core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _summarize_kv_update_slots(slot_mapping: jax.Array) -> tuple[int, int, int]:
    """Return valid/skipped/duplicate slot counts for profiling diagnostics."""
    slot_mapping_np = np.asarray(jax.device_get(slot_mapping))
    valid_slots = slot_mapping_np[slot_mapping_np >= 0]
    valid_tokens = int(valid_slots.size)
    skipped_tokens = int(slot_mapping_np.size - valid_tokens)
    duplicate_slots = int(valid_tokens - np.unique(valid_slots).size)
    return valid_tokens, skipped_tokens, duplicate_slots


def update_kv_cache(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Internal KV-cache update operator boundary.

    The current default backend is the reference scatter implementation. The
    indirection exists so a dedicated kernel can replace it later without
    changing the attention layer call site.
    """
    backend = _active_kv_update_backend()

    profiling_enabled = decode_step_profiling_enabled()
    can_measure = profiling_enabled and not _tree_has_tracer(
        (key, value, k_cache, v_cache, slot_mapping)
    )
    started_at = perf_counter() if can_measure else 0.0
    if backend == "compact_scatter":
        out = store_kv_to_cache_compact_scatter(
            key, value, k_cache, v_cache, slot_mapping,
        )
    elif backend == "sorted_compact_scatter":
        out = store_kv_to_cache_sorted_compact_scatter(
            key, value, k_cache, v_cache, slot_mapping,
        )
    else:
        out = store_kv_to_cache(key, value, k_cache, v_cache, slot_mapping)
    if can_measure:
        block_until_ready_tree(out)
        valid_tokens, skipped_tokens, duplicate_slots = _summarize_kv_update_slots(
            slot_mapping
        )
        record_kv_update_stats(
            seconds=perf_counter() - started_at,
            tokens=int(slot_mapping.shape[0]),
            valid_tokens=valid_tokens,
            skipped_tokens=skipped_tokens,
            duplicate_slots=duplicate_slots,
            backend=backend,
            measured=True,
        )
    return out


def update_kv_cache_int8(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    k_scale: jax.Array,
    v_scale: jax.Array,
    slot_mapping: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Internal INT8 KV-cache update operator boundary."""
    backend = _active_kv_update_backend()

    profiling_enabled = decode_step_profiling_enabled()
    can_measure = profiling_enabled and not _tree_has_tracer(
        (key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping)
    )
    started_at = perf_counter() if can_measure else 0.0
    if backend == "compact_scatter":
        out = store_kv_to_cache_int8_compact_scatter(
            key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
        )
    elif backend == "sorted_compact_scatter":
        out = store_kv_to_cache_int8_sorted_compact_scatter(
            key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
        )
    else:
        out = store_kv_to_cache_int8(
            key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
        )
    if can_measure:
        block_until_ready_tree(out)
        valid_tokens, skipped_tokens, duplicate_slots = _summarize_kv_update_slots(
            slot_mapping
        )
        record_kv_update_stats(
            seconds=perf_counter() - started_at,
            tokens=int(slot_mapping.shape[0]),
            valid_tokens=valid_tokens,
            skipped_tokens=skipped_tokens,
            duplicate_slots=duplicate_slots,
            backend=backend,
            measured=True,
        )
    return out


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
    safe_block_tables = jnp.clip(block_tables, min=0, max=k_cache.shape[0] - 1)
    
    # Use direct block indexing - more memory efficient
    # Gather blocks: [batch, max_blocks] -> [batch, max_blocks, block_size, heads, dim]
    gathered_k_blocks = k_cache[safe_block_tables]  # [batch, max_blocks, block_size, heads, dim]
    gathered_v_blocks = v_cache[safe_block_tables]
    
    # Reshape to [batch, max_context_len, heads, dim]
    gathered_k = gathered_k_blocks.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    gathered_v = gathered_v_blocks.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    
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
    implementation = _prefill_attn_impl_for_dtype(q.dtype)

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
            implementation=implementation,
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
    q_src_indices = jnp.clip(q_src_indices, min=0, max=total_tokens - 1)
    k_src_indices = jnp.clip(k_src_indices, min=0, max=total_tokens - 1)
    
    # Validity masks for padding
    q_valid = pos_indices_q < seq_lens_q[:, None]  # [batch, max_seq_q]
    k_valid = pos_indices_k < seq_lens_k[:, None]  # [batch, max_seq_k]
    
    # Vectorized gather using advanced indexing - O(1) parallel operation!
    # q: [total_tokens, heads, dim] -> q_padded: [batch, max_seq, heads, dim]
    q_padded = q[q_src_indices]  # [batch, max_seq_q, heads, dim]
    k_padded = k[k_src_indices]  # [batch, max_seq_k, heads, dim]
    v_padded = v[k_src_indices]  # [batch, max_seq_k, heads, dim]
    
    # Zero out invalid query positions. K/V invalid positions are already
    # excluded by the attention mask, so avoid extra write traffic there.
    q_padded = jnp.where(q_valid[:, :, None, None], q_padded, 0)
    
    # Create attention mask combining causal + padding
    # [batch, max_seq_q, max_seq_k]
    q_pos = jnp.arange(max_seqlen_q)[None, :, None]  # [1, seq_q, 1]
    k_pos = jnp.arange(max_seqlen_k)[None, None, :]  # [1, 1, seq_k]
    
    # Causal mask: q_pos >= k_pos
    causal_mask = q_pos >= k_pos  # [1, seq_q, seq_k]
    
    # Padding mask: only attend to valid positions (reuse precomputed masks)
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
        implementation=implementation,
    )  # [batch, max_seq_q, heads, dim]
    
    # ==========================================================================
    # VECTORIZED scatter back to packed output (replaces O(N) fori_loop)
    # ==========================================================================
    
    # Compute destination indices for each (batch, position) -> packed index
    # dest_indices[b, p] = cu_seqlens_q[b] + p
    dest_indices = cu_seqlens_q[:-1, None] + pos_indices_q  # [batch, max_seq_q]
    dest_indices = jnp.clip(dest_indices, min=0, max=total_tokens - 1)
    
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

    def set_kv_cache(
        self,
        k_cache: jax.Array,
        v_cache: jax.Array,
    ):
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
        # Use cache dtype (typically bfloat16) to enable cuDNN attention on GPU
        target_dtype = self.k_cache.dtype if self.k_cache is not None else q.dtype
        if q.dtype != target_dtype:
            q = q.astype(target_dtype)
        if k.dtype != target_dtype:
            k = k.astype(target_dtype)
        if v.dtype != target_dtype:
            v = v.astype(target_dtype)

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
        
        Uses Pallas paged attention kernel when available, which avoids
        gathering K/V into dense tensors (major performance improvement).
        
        Args:
            q: Query tensor [batch_size, num_heads, head_dim].
            context: Attention context with cache info.
        
        Returns:
            Output tensor [batch_size, num_heads, head_dim].
        """
        # Try to use Pallas paged attention kernel (avoids gather overhead)
        if PALLAS_AVAILABLE and pallas_paged_attention is not None:
            # Cast query to cache dtype for consistent ops
            target_dtype = self.k_cache.dtype if self.k_cache is not None else q.dtype
            if q.dtype != target_dtype:
                q = q.astype(target_dtype)
            
            return pallas_paged_attention(
                q=q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=context.block_tables,
                context_lens=context.context_lens,
                scale=self.scale,
                block_size=self.block_size,
                decode_schedule_token=context.decode_schedule_token,
            )
        
        # Fallback: Gather K/V from paged cache (slower path)
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
        if q.dtype != target_dtype:
            q = q.astype(target_dtype)
        if k_gathered.dtype != target_dtype:
            k_gathered = k_gathered.astype(target_dtype)
        if v_gathered.dtype != target_dtype:
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
            implementation="cudnn" if (jax.default_backend() == "gpu" and q.dtype == jnp.float16) else None,
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
            k_cache_flat = self.k_cache.reshape(cache_shape)
            v_cache_flat = self.v_cache.reshape(cache_shape)

            k_cache_flat, v_cache_flat = update_kv_cache(
                k, v, k_cache_flat, v_cache_flat, context.slot_mapping,
            )

            self.k_cache = k_cache_flat.reshape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
            )
            self.v_cache = v_cache_flat.reshape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
            )
        
        if context.is_prefill:
            return self._prefill_attention(q, k, v, context)
        else:
            return self._decode_attention(q, context)
