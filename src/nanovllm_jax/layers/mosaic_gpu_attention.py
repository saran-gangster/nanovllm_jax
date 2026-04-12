"""Mosaic GPU attention kernels for paged decode and prefill.

This module implements high-performance attention kernels using Mosaic GPU backend features:
- plgpu.SMEM for shared memory with TilingTransform and SwizzleTransform
- plgpu.wgmma for TensorCore matrix multiply-accumulate
- plgpu.Barrier for async pipeline coordination
- plgpu.emit_pipeline for memory/compute overlap
- plgpu.copy_gmem_to_smem / copy_smem_to_gmem for TMA transfers

Kernel families in this module:
- baseline: stable batched decode core
- latency: short-context partitioned decode path
- throughput: long-context split-k decode path
- throughput_v2: row-native partitioned decode partials + device reduction

Key algorithms:
- online softmax with log2/exp2 for FMA utilization
- Warp specialization: 2 compute warpgroups + 1 memory warpgroup
- Paged KV-cache with block table indirection

Reference implementations:
- good_bowl/docs/reference/attention_mgpu.py - FlashAttention3 forward/backward
- good_bowl/docs/reference/ragged_dot_mgpu.py - Variable-length group handling
- good_bowl/docs/reference/hopper_matmul_mgpu.py - WGMMA pipeline pattern

Constraints:
- WGMMA requires M dimension >= 64 (batch queries across sequences for decode)
- Block sizes must be multiples of 64 for WGMMA alignment
- SMEM limited to ~228KB on H100

Author: nanovllm_jax
"""

import dataclasses
import functools
import json
import math
import os
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax import lax
try:
    from jax.extend import backend as jax_backend
except ImportError:
    jax_backend = None

from nanovllm_jax.utils.runtime_diagnostics import (
    block_until_ready_tree,
    decode_step_profiling_enabled,
    record_partitioned_decode_reduction_stats,
)

# Check if Pallas Mosaic GPU is available
try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu
    MOSAIC_AVAILABLE = True
    _WGMMA_ROW = getattr(plgpu.Layout, "WGMMA_ROW", plgpu.Layout.WGMMA.reduce(1))
except ImportError:
    MOSAIC_AVAILABLE = False
    pl = None
    plgpu = None
    _WGMMA_ROW = None


def _check_mosaic_available():
    """Check if Pallas Mosaic GPU backend is available."""
    if not MOSAIC_AVAILABLE:
        raise RuntimeError(
            "Pallas Mosaic GPU backend is not available. "
            "Requires JAX with GPU support (jax[cuda12] or jax[cuda11])."
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MosaicAttentionConfig:
    """Configuration for Mosaic GPU paged attention kernels.
    
    Attributes:
        block_q: Query block size (must be multiple of 64 for WGMMA).
        block_kv: KV block size for tiling (must be multiple of 64).
        max_concurrent_steps: Pipeline depth (2-4 typically).
        use_schedule_barrier: Enable TensorCore coordination barriers.
        num_compute_wgs: Number of compute warpgroups (typically 1-2).
    """
    block_q: int = 64       # Query tile size (M dimension for WGMMA)
    block_kv: int = 64      # KV tile size
    max_concurrent_steps: int = 2  # Pipeline depth
    use_schedule_barrier: bool = True
    num_compute_wgs: int = 2  # 2 compute + 1 memory = 3 total
    
    def __post_init__(self):
        if self.block_q % 64 != 0:
            raise ValueError(f"block_q={self.block_q} must be multiple of 64 for WGMMA")
        if self.block_kv % 64 != 0:
            raise ValueError(f"block_kv={self.block_kv} must be multiple of 64 for WGMMA")
        if self.max_concurrent_steps < 2:
            raise ValueError(f"max_concurrent_steps={self.max_concurrent_steps} must be >= 2")


@dataclasses.dataclass(frozen=True)
class ThroughputV2Plan:
    """Schedule-owned throughput-v2 execution plan.

    The plan is intentionally small so it can live in the per-family decode
    schedule cache without inheriting the old flattened tile-chunk metadata
    model.
    """

    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_blocks_per_seq: int
    block_size: int
    block_kv: int
    q_heads_per_kv_head: int
    k_splits: int
    pages_per_partition: int
    max_concurrent_steps: int
    num_compute_wgs: int
    use_schedule_barrier: bool
    partial_kernel: str
    launch_block_q: int | None
    launch_max_concurrent_steps: int | None
    launch_num_compute_wgs: int | None
    launch_num_memory_wgs: int | None
    launch_use_schedule_barrier: bool | None
    uses_wrapper_partitioning: bool
    uses_batched_core: bool
    reduction_boundary: str
    reduction_backend: str
    metadata_model: str
    metadata_cache_key: tuple[object, ...]


_SMEM_BUDGET_BYTES = 232_448  # 228 KB on Hopper (in bytes)
_METADATA_ALIGNMENT = 128
_BARRIER_BYTES_PER_SLOT = 16  # heuristic: small allowance per barrier slot
_THROUGHPUT_SPLITK_TABLE_PATH = (
    os.environ.get("NANOVLLM_JAX_MOSAIC_THROUGHPUT_SPLITK_TABLE_PATH", "").strip() or None
)
_THROUGHPUT_SPLITK_TABLE: dict[tuple[int, int, int, int], int] | None = None
_PARTITIONED_DECODE_REDUCTION_BACKEND = os.environ.get(
    "NANOVLLM_JAX_PARTITIONED_DECODE_REDUCTION_BACKEND", "streaming"
).strip().lower()


def _parse_shape_key(raw_key: str) -> tuple[int, int, int, int] | None:
    fields: dict[str, int] = {}
    try:
        for token in raw_key.split(","):
            if "=" not in token:
                return None
            key, value = token.split("=", 1)
            fields[key.strip()] = int(value.strip())
        return (
            int(fields["batch"]),
            int(fields["head_dim"]),
            int(fields["blocks"]),
            int(fields["block_size"]),
        )
    except Exception:
        return None


def _get_or_prepare_cached_metadata(
    prepared_metadata_cache: dict[tuple, object] | None,
    key: tuple,
    factory,
):
    if prepared_metadata_cache is None:
        return factory()
    metadata = prepared_metadata_cache.get(key)
    if metadata is None:
        metadata = factory()
        prepared_metadata_cache[key] = metadata
    return metadata


def _normalize_partitioned_decode_reduction_backend(raw: str) -> str:
    backend = str(raw).strip().lower()
    if backend in {"auto", "streaming", "device"}:
        return backend
    raise ValueError(
        "NANOVLLM_JAX_PARTITIONED_DECODE_REDUCTION_BACKEND must be one of: "
        "auto|streaming|device"
    )


def _active_partitioned_decode_reduction_backend(backend_override: str | None = None) -> str:
    source = _PARTITIONED_DECODE_REDUCTION_BACKEND if backend_override is None else backend_override
    backend = _normalize_partitioned_decode_reduction_backend(source)
    if backend == "auto":
        return "streaming"
    return backend


def _tree_has_tracer(value) -> bool:
    return any(
        isinstance(leaf, jax_core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def configure_throughput_splitk_table(path: str | None) -> None:
    """Set optional split-k override table path (loaded lazily)."""
    global _THROUGHPUT_SPLITK_TABLE_PATH, _THROUGHPUT_SPLITK_TABLE
    _THROUGHPUT_SPLITK_TABLE_PATH = str(path).strip() if path is not None else ""
    _THROUGHPUT_SPLITK_TABLE_PATH = _THROUGHPUT_SPLITK_TABLE_PATH or None
    _THROUGHPUT_SPLITK_TABLE = None


def _load_throughput_splitk_table_if_needed() -> None:
    global _THROUGHPUT_SPLITK_TABLE
    if _THROUGHPUT_SPLITK_TABLE is not None:
        return
    parsed: dict[tuple[int, int, int, int], int] = {}
    if _THROUGHPUT_SPLITK_TABLE_PATH is not None:
        try:
            payload = json.loads(Path(_THROUGHPUT_SPLITK_TABLE_PATH).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for raw_key, raw_value in payload.items():
                    if not isinstance(raw_key, str):
                        continue
                    shape_key = _parse_shape_key(raw_key)
                    if shape_key is None:
                        continue
                    split_k = int(raw_value)
                    if split_k >= 1:
                        parsed[shape_key] = split_k
        except Exception:
            parsed = {}
    _THROUGHPUT_SPLITK_TABLE = parsed


def _lookup_throughput_splitk_override(
    *,
    batch_size: int,
    head_dim: int,
    max_blocks_per_seq: int,
    block_size: int,
) -> int | None:
    _load_throughput_splitk_table_if_needed()
    if _THROUGHPUT_SPLITK_TABLE is None:
        return None
    key = (batch_size, head_dim, max_blocks_per_seq, block_size)
    return _THROUGHPUT_SPLITK_TABLE.get(key)


def _pad_last_dim_to_multiple(
    arr: jax.Array,
    multiple: int = _METADATA_ALIGNMENT,
    pad_value: int | float = 0,
):
    """Right-pad the last dimension so Mosaic layout casts meet alignment rules."""

    last_dim = arr.shape[-1]
    pad = (-last_dim) % multiple
    if pad == 0:
        return arr
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, pad)]
    return jnp.pad(arr, pad_width, constant_values=pad_value)


def _cap_pipeline_depth(
    *,
    block_q: int,
    block_kv: int,
    head_dim: int,
    dtype,
    num_compute_wgs: int,
    requested_steps: int,
    qo_rows_per_wg: int | None = None,
    extra_smem_bytes: int = 0,
    include_barrier_overhead: bool = True,
    metadata_bytes: int = 0,
):
    """Clamp max_concurrent_steps so SMEM usage never exceeds the Hopper budget."""

    dtype_size = jnp.dtype(dtype).itemsize
    qo_rows = block_q if qo_rows_per_wg is None else qo_rows_per_wg
    base_qo_bytes = num_compute_wgs * qo_rows * head_dim * dtype_size * 2  # Q + O tiles

    barrier_bytes = 0
    if include_barrier_overhead:
        barrier_slots = 4 * requested_steps + num_compute_wgs + 1  # k,v,q,consumed,schedule
        barrier_bytes = barrier_slots * _BARRIER_BYTES_PER_SLOT

    base_bytes = base_qo_bytes + extra_smem_bytes + barrier_bytes + metadata_bytes
    available = _SMEM_BUDGET_BYTES - base_bytes
    if available <= 0:
        raise ValueError(
            "Mosaic kernel configuration exhausts shared memory before staging K/V. "
            "Reduce block_q, head_dim, or the number of compute warpgroups."
        )

    per_step_bytes = block_kv * head_dim * dtype_size * 2  # K + V per pipeline stage
    if per_step_bytes <= 0:
        raise ValueError("block_kv and head_dim must be positive for Mosaic kernels")

    if available < per_step_bytes:
        raise ValueError(
            "Insufficient shared memory for a single KV tile. "
            "Decrease block_kv or head_dim to proceed with Mosaic decode."
        )

    max_steps_possible = available // per_step_bytes
    effective_steps = max(1, min(requested_steps, max_steps_possible))
    return effective_steps


def _decode_compiler_params(*, throughput_mode: bool):
    """Create compiler params for decode kernels with throughput-safe fallback."""

    if throughput_mode:
        try:
            return plgpu.CompilerParams(
                approx_math=True, unsafe_no_auto_barriers=True,
            )
        except TypeError:
            # Older JAX versions may not expose unsafe_no_auto_barriers.
            pass
    return plgpu.CompilerParams(approx_math=True)


def _default_core_count() -> int:
    """Best-effort estimate of available SMs/cores for split-k heuristics."""

    if jax_backend is not None:
        try:
            return int(jax_backend.get_default_device().core_count)
        except Exception:
            pass
    return max(1, len(jax.devices()))


# =============================================================================
# Decode Utility Helpers
# =============================================================================

class MosaicDecodeMetadata(NamedTuple):
    """Pre-computed tile metadata for Mosaic decode (shareable across layers).

    The metadata depends only on (block_tables, context_lens, block_q, block_size,
    block_kv) — NOT on q.  Computing it once per step and sharing across all
    attention layers avoids redundant work.
    """
    block_tables: jax.Array          # padded
    context_lens: jax.Array          # padded
    tile_valid_counts: jax.Array
    tile_row_offsets: jax.Array
    tile_row_lengths: jax.Array
    tile_chunk_block_indices: jax.Array
    tile_chunk_offsets: jax.Array
    tile_chunk_flat_positions: jax.Array
    tile_chunk_tokens: jax.Array
    tile_chunk_prefix_tokens: jax.Array
    tile_chunk_row_indices: jax.Array
    tile_chunk_logical_blocks: jax.Array
    tile_chunk_counts: jax.Array
    original_batch_size: int
    pad_rows: int


def _select_decode_num_compute_wgs(
    *,
    requested_num_compute_wgs: int,
    block_q: int,
    batch_size: int,
) -> int:
    """Pick compute WG count for decode while preserving WGMMA constraints."""

    requested = max(1, min(int(requested_num_compute_wgs), 2))
    if requested == 1:
        return 1
    if block_q % requested != 0:
        return 1
    rows_per_wg = block_q // requested
    if rows_per_wg < 64:
        return 1
    if batch_size < block_q:
        return 1
    return requested


def prepare_decode_metadata(
    block_tables: jax.Array,
    context_lens: jax.Array,
    batch_size: int,
    block_q: int,
    block_size: int,
    block_kv: int,
    include_unused_fields: bool = True,
) -> MosaicDecodeMetadata:
    """Build per-tile chunk schedules for WGMMA decode kernels.

    This is the expensive part of ``_prepare_decode_tiles`` — the q-padding
    is trivial and handled separately.  Callers should cache the returned
    ``MosaicDecodeMetadata`` across layers within a single decode step.
    """

    if block_size % block_kv:
        raise ValueError("block_size must be divisible by block_kv for chunk scheduling")

    original_batch_size = batch_size
    pad_rows = (-original_batch_size) % block_q

    if pad_rows:
        block_tables = jnp.pad(block_tables, ((0, pad_rows), (0, 0)), constant_values=0)
        context_lens = jnp.pad(context_lens, ((0, pad_rows),), constant_values=0)

    padded_batch = original_batch_size + pad_rows
    num_batch_tiles = padded_batch // block_q
    tile_valid_counts = jnp.full((num_batch_tiles,), block_q, dtype=jnp.int32)
    if pad_rows:
        tile_valid_counts = tile_valid_counts.at[-1].set(block_q - pad_rows)

    max_blocks = block_tables.shape[1]
    small_context_fast_path = max_blocks <= 12
    chunks_per_block = block_size // block_kv
    max_tile_chunks = block_q * max_blocks * chunks_per_block

    block_tables_tile = block_tables.reshape(num_batch_tiles, block_q, max_blocks)
    context_tile = context_lens.reshape(num_batch_tiles, block_q)

    block_offsets = jnp.arange(max_blocks, dtype=jnp.int32) * block_size
    block_offsets = block_offsets[None, None, :]
    tokens_per_block = jnp.clip(
        context_tile[:, :, None] - block_offsets,
        min=0,
        max=block_size,
    )

    chunk_offsets = jnp.arange(chunks_per_block, dtype=jnp.int32) * block_kv
    chunk_offsets = chunk_offsets[None, None, None, :]
    tokens_per_chunk = jnp.clip(
        tokens_per_block[:, :, :, None] - chunk_offsets,
        min=0,
        max=block_kv,
    )

    chunk_flat_positions = (
        block_tables_tile[:, :, :, None].astype(jnp.int32) * block_size
        + chunk_offsets
    )
    chunk_offsets = jnp.broadcast_to(chunk_offsets, tokens_per_chunk.shape)

    chunk_flat_positions = chunk_flat_positions.reshape(num_batch_tiles, max_tile_chunks)
    chunk_tokens = tokens_per_chunk.reshape(num_batch_tiles, max_tile_chunks)

    row_ids = jnp.arange(block_q, dtype=jnp.int32)
    row_ids = row_ids[None, :, None, None]
    row_ids = jnp.broadcast_to(row_ids, tokens_per_chunk.shape)
    chunk_row_indices = row_ids.reshape(num_batch_tiles, max_tile_chunks)

    chunk_valid = chunk_tokens > 0
    tile_chunk_counts = chunk_valid.sum(axis=1, dtype=jnp.int32)
    chunk_tokens = chunk_tokens.astype(jnp.int32)
    chunk_offsets = chunk_offsets.reshape(num_batch_tiles, max_tile_chunks).astype(jnp.int32)

    # Compact valid chunks to the front of each tile's schedule using
    # prefix-sum + scatter (O(n) vs O(n log n) argsort).
    valid_i32 = chunk_valid.astype(jnp.int32)
    invalid_i32 = 1 - valid_i32
    valid_dest = jnp.cumsum(valid_i32, axis=1) - 1
    invalid_dest = tile_chunk_counts[:, None] + jnp.cumsum(invalid_i32, axis=1) - 1
    dest = jnp.where(chunk_valid, valid_dest, invalid_dest)

    src = jnp.broadcast_to(
        jnp.arange(max_tile_chunks, dtype=jnp.int32)[None, :],
        (num_batch_tiles, max_tile_chunks),
    )
    row = jnp.broadcast_to(
        jnp.arange(num_batch_tiles, dtype=jnp.int32)[:, None],
        (num_batch_tiles, max_tile_chunks),
    )
    order = jnp.zeros((num_batch_tiles, max_tile_chunks), dtype=jnp.int32)
    order = order.at[row, dest].set(src)

    chunk_flat_positions = jnp.take_along_axis(chunk_flat_positions, order, axis=1)
    chunk_tokens = jnp.take_along_axis(chunk_tokens, order, axis=1)
    chunk_row_indices = jnp.take_along_axis(chunk_row_indices, order, axis=1)

    if include_unused_fields:
        chunk_block_indices = jnp.broadcast_to(
            block_tables_tile[:, :, :, None], tokens_per_chunk.shape
        ).reshape(num_batch_tiles, max_tile_chunks).astype(jnp.int32)
        chunk_block_indices = jnp.take_along_axis(chunk_block_indices, order, axis=1)
        chunk_offsets = jnp.take_along_axis(chunk_offsets, order, axis=1)
        row_lengths = context_tile.astype(jnp.int32)
        row_offsets = jnp.cumsum(row_lengths, axis=1, dtype=jnp.int32) - row_lengths

        if small_context_fast_path:
            # Decode kernel currently consumes row_indices/tokens/flat_positions only.
            # For short contexts, avoid expensive prefix/logical metadata construction.
            chunk_prefix = jnp.zeros_like(chunk_tokens, dtype=jnp.int32)
            chunk_logical_blocks = jnp.zeros_like(chunk_tokens, dtype=jnp.int32)
        else:
            logical_ids = jnp.arange(max_blocks, dtype=jnp.int32)
            logical_ids = logical_ids[None, None, :, None]
            logical_ids = jnp.broadcast_to(logical_ids, tokens_per_chunk.shape)
            chunk_logical_blocks = logical_ids.reshape(num_batch_tiles, max_tile_chunks)
            chunk_logical_blocks = jnp.take_along_axis(chunk_logical_blocks, order, axis=1)

            tokens_per_row = tokens_per_chunk.reshape(num_batch_tiles, block_q, -1).astype(jnp.int32)
            chunk_prefix = jnp.cumsum(tokens_per_row, axis=2, dtype=jnp.int32) - tokens_per_row
            chunk_prefix = chunk_prefix.reshape(num_batch_tiles, max_tile_chunks)
            chunk_prefix = jnp.take_along_axis(chunk_prefix, order, axis=1)

        tile_row_offsets = _pad_last_dim_to_multiple(row_offsets)
        tile_row_lengths = _pad_last_dim_to_multiple(row_lengths)
        tile_chunk_block_indices = _pad_last_dim_to_multiple(chunk_block_indices)
        tile_chunk_offsets = _pad_last_dim_to_multiple(chunk_offsets)
        tile_chunk_prefix_tokens = _pad_last_dim_to_multiple(chunk_prefix)
        tile_chunk_logical_blocks = _pad_last_dim_to_multiple(chunk_logical_blocks)
    else:
        # Runtime decode kernel does not consume these tensors; keep compact
        # placeholders to reduce metadata prep and argument footprint.
        tile_row_offsets = jnp.zeros((num_batch_tiles, 1), dtype=jnp.int32)
        tile_row_lengths = jnp.zeros((num_batch_tiles, 1), dtype=jnp.int32)
        tile_chunk_block_indices = jnp.zeros((num_batch_tiles, 1), dtype=jnp.int32)
        tile_chunk_offsets = jnp.zeros((num_batch_tiles, 1), dtype=jnp.int32)
        tile_chunk_prefix_tokens = jnp.zeros((num_batch_tiles, 1), dtype=jnp.int32)
        tile_chunk_logical_blocks = jnp.zeros((num_batch_tiles, 1), dtype=jnp.int32)

    # Pad metadata consumed by decode kernel so Mosaic layout requirements are
    # satisfied (multiples of 128).
    tile_chunk_flat_positions = _pad_last_dim_to_multiple(chunk_flat_positions)
    tile_chunk_tokens = _pad_last_dim_to_multiple(chunk_tokens)
    tile_chunk_row_indices = _pad_last_dim_to_multiple(chunk_row_indices)

    return MosaicDecodeMetadata(
        block_tables=block_tables,
        context_lens=context_lens,
        tile_valid_counts=tile_valid_counts,
        tile_row_offsets=tile_row_offsets,
        tile_row_lengths=tile_row_lengths,
        tile_chunk_block_indices=tile_chunk_block_indices,
        tile_chunk_offsets=tile_chunk_offsets,
        tile_chunk_flat_positions=tile_chunk_flat_positions,
        tile_chunk_tokens=tile_chunk_tokens,
        tile_chunk_prefix_tokens=tile_chunk_prefix_tokens,
        tile_chunk_row_indices=tile_chunk_row_indices,
        tile_chunk_logical_blocks=tile_chunk_logical_blocks,
        tile_chunk_counts=tile_chunk_counts,
        original_batch_size=original_batch_size,
        pad_rows=pad_rows,
    )


def _prepare_decode_tiles(
    q: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    block_q: int,
    block_size: int,
    block_kv: int,
    metadata: MosaicDecodeMetadata | None = None,
):
    """Pad decode inputs and build per-tile chunk schedules for WGMMA kernels.

    When *metadata* is provided (pre-computed via ``prepare_decode_metadata``),
    only the trivial q-padding is performed.  Otherwise the full metadata is
    computed inline (baseline path).
    """

    if metadata is None:
        metadata = prepare_decode_metadata(
            block_tables,
            context_lens,
            q.shape[0],
            block_q,
            block_size,
            block_kv,
            include_unused_fields=False,
        )

    pad_rows = metadata.pad_rows
    if pad_rows:
        q = jnp.pad(q, ((0, pad_rows), (0, 0), (0, 0)))

    return (
        q,
        metadata.block_tables,
        metadata.context_lens,
        metadata.tile_valid_counts,
        metadata.tile_row_offsets,
        metadata.tile_row_lengths,
        metadata.tile_chunk_block_indices,
        metadata.tile_chunk_offsets,
        metadata.tile_chunk_flat_positions,
        metadata.tile_chunk_tokens,
        metadata.tile_chunk_prefix_tokens,
        metadata.tile_chunk_row_indices,
        metadata.tile_chunk_logical_blocks,
        metadata.tile_chunk_counts,
        metadata.original_batch_size,
    )


# =============================================================================
# Prefill Helpers (GroupInfo-style metadata)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PrefillTileInfo:
    """Per-tile metadata for ragged prefill sequences (similar to GroupInfo)."""

    block_start: jax.Array          # Tile start index (relative to sequence)
    actual_start: jax.Array         # First valid token inside this tile
    actual_end: jax.Array           # Exclusive end of valid tokens
    start_within_block: jax.Array   # Offset of valid region within tile
    actual_size: jax.Array          # Number of valid rows in tile

    @classmethod
    def create(cls, seq_len: jax.Array, tile_size: int, tile_idx: jax.Array):
        tile_idx = tile_idx.astype(jnp.int32)
        tile_size = jnp.int32(tile_size)
        block_start = tile_idx * tile_size
        block_end = block_start + tile_size
        actual_start = jnp.minimum(block_start, seq_len)
        actual_end = jnp.minimum(block_end, seq_len)
        actual_size = jnp.maximum(actual_end - actual_start, 0)
        start_within = jnp.maximum(actual_start - block_start, 0)
        return cls(
            block_start=block_start,
            actual_start=actual_start,
            actual_end=actual_end,
            start_within_block=start_within,
            actual_size=actual_size,
        )


# =============================================================================
# SMEM Layout Helpers
# =============================================================================

def get_smem_transforms(tile_k: int, dtype):
    """Compute optimal swizzle and tiling transforms for SMEM.
    
    Args:
        tile_k: The K dimension of the tile (for computing swizzle).
        dtype: Data type for size calculation.
    
    Returns:
        Tuple of (TilingTransform, SwizzleTransform).
    """
    # Find optimal swizzle (128-byte is common for avoiding bank conflicts)
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    
    # Standard (8, swizzle_elems) tiling for WGMMA compatibility
    tiling = plgpu.TilingTransform((8, swizzle_elems))
    swizzle_transform = plgpu.SwizzleTransform(swizzle)
    
    return (tiling, swizzle_transform)


# =============================================================================
# Batched Decode Attention Kernel (WGMMA-compatible)
# =============================================================================

def batched_decode_attention_mosaic(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    config: MosaicAttentionConfig,
    metadata: MosaicDecodeMetadata | None = None,
    return_partials: bool = False,
    throughput_mode: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """Batched decode attention using true Mosaic GPU primitives.

    For decode phase, we batch 64+ sequences together to satisfy WGMMA M>=64 constraint.
    Each kernel invocation processes block_q sequences simultaneously.

    Algorithm (FlashAttention3 style):
    1. Load Q block (block_q queries) to SMEM
    2. For each KV block in the sequence:
       a. TMA load K, V blocks to SMEM (pipelined)
       b. WGMMA: QK^T -> scores
       c. Online softmax with log2/exp2
       d. WGMMA: P @ V -> accumulator update
    3. Normalize and store output

    Args:
        q: Query tensor [batch_size, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: Block indices [batch_size, max_blocks_per_seq].
        context_lens: Context lengths [batch_size].
        scale: Softmax scale (1/sqrt(head_dim)).
        config: Kernel configuration.
        metadata: Optional pre-computed tile metadata (from ``prepare_decode_metadata``).
            When provided, avoids redundant metadata computation across layers.
        throughput_mode: Enable long-context throughput scheduling tweaks
            (manual-barrier compiler
            mode and softmax barrier ordering).

    Returns:
        When ``return_partials`` is False (default):
          Output tensor [batch_size, num_heads, head_dim].
        When ``return_partials`` is True:
          Tuple ``(acc_partial, l_partial, m_partial)`` where
          ``acc_partial`` is unnormalized FP32 numerator, and ``l_partial`` /
          ``m_partial`` are online-softmax stats in log2 space.
    """
    _check_mosaic_available()

    # Extract shapes first (needed for _prepare_decode_tiles)
    batch_size, num_heads, head_dim = q.shape
    num_kv_blocks, kv_block_size, num_kv_heads, _ = k_cache.shape

    block_q = config.block_q
    block_kv = config.block_kv
    requested_steps = config.max_concurrent_steps

    # Decode can use two compute warpgroups when each WG still satisfies
    # WGMMA's M >= 64 constraint. Keep a 1-WG fallback for smaller batches.
    num_compute_wgs = _select_decode_num_compute_wgs(
        requested_num_compute_wgs=config.num_compute_wgs,
        block_q=block_q,
        batch_size=batch_size,
    )
    rows_per_wg = block_q // num_compute_wgs

    if block_q % num_compute_wgs != 0:
        raise ValueError(
            f"block_q={block_q} must be divisible by num_compute_wgs={num_compute_wgs}"
        )
    if rows_per_wg < 64:
        raise ValueError(
            f"rows_per_wg={rows_per_wg} must be >= 64 for WGMMA decode."
        )

    # Schedule barriers are only needed when multiple compute WGs are active.
    use_schedule_barrier = config.use_schedule_barrier and (num_compute_wgs > 1)

    if kv_block_size % block_kv != 0:
        raise ValueError(
            "KV cache block_size must be divisible by block_kv for Mosaic decode. "
            f"Received block_size={kv_block_size}, block_kv={block_kv}."
        )

    # Decode issues one token per sequence, so we batch block_q sequences together
    # (block_q ≥ 64) to satisfy WGMMA M dimension requirements.
    (
        q,
        block_tables,
        context_lens,
        tile_valid_counts,
        tile_row_offsets,
        tile_row_lengths,
        tile_chunk_block_indices,
        tile_chunk_offsets,
        tile_chunk_flat_positions,
        tile_chunk_tokens,
        tile_chunk_prefix_tokens,
        tile_chunk_row_indices,
        tile_chunk_logical_blocks,
        tile_chunk_counts,
        original_batch_size,
    ) = _prepare_decode_tiles(
        q,
        block_tables,
        context_lens,
        block_q,
        kv_block_size,
        block_kv,
        metadata=metadata,
    )

    # Re-extract shapes after padding
    batch_size, num_heads, head_dim = q.shape
    max_blocks_per_seq = block_tables.shape[1]

    # Per-tile metadata dimensions for SMEM preload
    padded_max_tile_chunks = tile_chunk_row_indices.shape[1]

    # TMA async copies support at most 256 elements per dimension.
    # When padded_max_tile_chunks exceeds this limit, fall back to GMEM reads.
    use_smem_metadata = (padded_max_tile_chunks <= 256)

    # SMEM budget for metadata buffers (3 arrays preloaded per tile):
    # row_indices, tokens, flat_positions.
    num_meta_smem_arrays = 3
    if use_smem_metadata:
        meta_bytes = (
            num_meta_smem_arrays * padded_max_tile_chunks * jnp.dtype(jnp.int32).itemsize
            + 2 * _BARRIER_BYTES_PER_SLOT  # compute_meta + memory_meta barriers
        )
    else:
        meta_bytes = 0

    max_concurrent_steps = _cap_pipeline_depth(
        block_q=block_q,
        block_kv=block_kv,
        head_dim=head_dim,
        dtype=q.dtype,
        num_compute_wgs=num_compute_wgs,
        requested_steps=requested_steps,
        qo_rows_per_wg=rows_per_wg,
        metadata_bytes=meta_bytes,
    )
    
    # GQA: query heads per KV head
    q_heads_per_kv_head = num_heads // num_kv_heads
    
    # For decode, we process block_q sequences at a time
    # This ensures M >= 64 for WGMMA
    num_batch_tiles = (batch_size + block_q - 1) // block_q
    
    # Compute SMEM transforms
    transforms = get_smem_transforms(head_dim, q.dtype)
    
    # Maximum KV steps across ALL rows in a tile.
    # The chunk schedule flattens every row's chunks into one sequence:
    # block_q rows × max_blocks × chunks_per_block.
    chunks_per_block = kv_block_size // block_kv
    max_kv_steps = block_q * max_blocks_per_seq * chunks_per_block
    
    def _kernel_entry_impl(
        q_ref,
        k_cache_flat_ref,
        v_cache_flat_ref,
        tile_valid_counts_ref,
        tile_row_offsets_ref,
        tile_row_lengths_ref,
        tile_chunk_block_indices_ref,
        tile_chunk_offsets_ref,
        tile_chunk_flat_positions_ref,
        tile_chunk_tokens_ref,
        tile_chunk_prefix_tokens_ref,
        tile_chunk_row_indices_ref,
        tile_chunk_logical_blocks_ref,
        tile_chunk_counts_ref,
        out_refs,
    ):
        """Kernel entry point that allocates SMEM and barriers."""
        
        # Allocate SMEM buffers with swizzle for bank-conflict-free access
        # Q/O buffers: one [rows_per_wg, head_dim] tile per compute WG.
        qo_smem = plgpu.SMEM(
            (num_compute_wgs, rows_per_wg, head_dim),
            q.dtype,
            transforms=transforms,
        )
        
        # K buffer: [max_concurrent_steps, block_kv, head_dim]
        k_smem = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            q.dtype,
            transforms=transforms,
        )
        
        # V buffer: [max_concurrent_steps, block_kv, head_dim]
        v_smem = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            q.dtype,
            transforms=transforms,
        )
        
        # Barriers for async pipeline
        k_barriers = plgpu.Barrier(num_barriers=max_concurrent_steps)
        v_barriers = plgpu.Barrier(num_barriers=max_concurrent_steps)
        q_barriers = plgpu.Barrier(num_barriers=num_compute_wgs)
        
        # Consumed barriers (multiple arrivals from compute warpgroups)
        k_consumed = plgpu.Barrier(
            num_arrivals=num_compute_wgs,
            num_barriers=max_concurrent_steps,
        )
        v_consumed = plgpu.Barrier(
            num_arrivals=num_compute_wgs,
            num_barriers=max_concurrent_steps,
        )
        
        # Schedule barrier for TensorCore coordination
        schedule_barrier = plgpu.Barrier(num_arrivals=num_compute_wgs)

        # Metadata SMEM buffers (preloaded once per tile via TMA).
        # When padded_max_tile_chunks > 256, TMA cannot handle the copy,
        # so we allocate minimal dummy SMEM and read metadata from GMEM.
        meta_smem_size = padded_max_tile_chunks if use_smem_metadata else 1
        chunk_row_indices_smem = plgpu.SMEM(
            (meta_smem_size,), jnp.int32,
        )
        chunk_tokens_smem = plgpu.SMEM(
            (meta_smem_size,), jnp.int32,
        )
        chunk_flat_positions_smem = plgpu.SMEM(
            (meta_smem_size,), jnp.int32,
        )

        # Metadata barriers (TMA arrivals)
        compute_meta_barrier = plgpu.Barrier(num_arrivals=2)  # row_indices + tokens
        memory_meta_barrier = plgpu.Barrier(num_arrivals=1)

        # Run kernel with scoped allocations
        pl.run_scoped(
            lambda *args: kernel_body(
                q_ref,
                k_cache_flat_ref,
                v_cache_flat_ref,
                tile_valid_counts_ref,
                tile_row_offsets_ref,
                tile_row_lengths_ref,
                tile_chunk_block_indices_ref,
                tile_chunk_offsets_ref,
                tile_chunk_flat_positions_ref,
                tile_chunk_tokens_ref,
                tile_chunk_prefix_tokens_ref,
                tile_chunk_row_indices_ref,
                tile_chunk_logical_blocks_ref,
                tile_chunk_counts_ref,
                out_refs,
                args,
            ),
            (qo_smem, k_smem, v_smem),  # SMEM buffers
            (k_barriers, v_barriers, q_barriers),  # Buffer barriers
            (k_consumed, v_consumed),  # Consumed barriers
            schedule_barrier,  # Schedule barrier
            (chunk_row_indices_smem, chunk_tokens_smem,
             chunk_flat_positions_smem),  # Meta SMEM
            (compute_meta_barrier, memory_meta_barrier),  # Meta barriers
            collective_axes="wg",
        )

    def kernel_entry(
        q_ref,
        k_cache_flat_ref,
        v_cache_flat_ref,
        tile_valid_counts_ref,
        tile_row_offsets_ref,
        tile_row_lengths_ref,
        tile_chunk_block_indices_ref,
        tile_chunk_offsets_ref,
        tile_chunk_flat_positions_ref,
        tile_chunk_tokens_ref,
        tile_chunk_prefix_tokens_ref,
        tile_chunk_row_indices_ref,
        tile_chunk_logical_blocks_ref,
        tile_chunk_counts_ref,
        out_ref,
    ):
        _kernel_entry_impl(
            q_ref,
            k_cache_flat_ref,
            v_cache_flat_ref,
            tile_valid_counts_ref,
            tile_row_offsets_ref,
            tile_row_lengths_ref,
            tile_chunk_block_indices_ref,
            tile_chunk_offsets_ref,
            tile_chunk_flat_positions_ref,
            tile_chunk_tokens_ref,
            tile_chunk_prefix_tokens_ref,
            tile_chunk_row_indices_ref,
            tile_chunk_logical_blocks_ref,
            tile_chunk_counts_ref,
            (out_ref,),
        )

    def kernel_entry_partials(
        q_ref,
        k_cache_flat_ref,
        v_cache_flat_ref,
        tile_valid_counts_ref,
        tile_row_offsets_ref,
        tile_row_lengths_ref,
        tile_chunk_block_indices_ref,
        tile_chunk_offsets_ref,
        tile_chunk_flat_positions_ref,
        tile_chunk_tokens_ref,
        tile_chunk_prefix_tokens_ref,
        tile_chunk_row_indices_ref,
        tile_chunk_logical_blocks_ref,
        tile_chunk_counts_ref,
        acc_out_ref,
        l_out_ref,
        m_out_ref,
    ):
        _kernel_entry_impl(
            q_ref,
            k_cache_flat_ref,
            v_cache_flat_ref,
            tile_valid_counts_ref,
            tile_row_offsets_ref,
            tile_row_lengths_ref,
            tile_chunk_block_indices_ref,
            tile_chunk_offsets_ref,
            tile_chunk_flat_positions_ref,
            tile_chunk_tokens_ref,
            tile_chunk_prefix_tokens_ref,
            tile_chunk_row_indices_ref,
            tile_chunk_logical_blocks_ref,
            tile_chunk_counts_ref,
            (acc_out_ref, l_out_ref, m_out_ref),
        )
    
    def kernel_body(
        q_ref,
        k_cache_flat_ref,
        v_cache_flat_ref,
        tile_valid_counts_ref,
        tile_row_offsets_ref,
        tile_row_lengths_ref,
        tile_chunk_block_indices_ref,
        tile_chunk_offsets_ref,
        tile_chunk_flat_positions_ref,
        tile_chunk_tokens_ref,
        tile_chunk_prefix_tokens_ref,
        tile_chunk_row_indices_ref,
        tile_chunk_logical_blocks_ref,
        tile_chunk_counts_ref,
        out_refs,
        scoped,
    ):
        """Main kernel body with warp specialization."""
        (smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier,
         meta_smem, meta_barriers) = scoped
        qo_smem, k_smem, v_smem = smem_buffers
        k_barriers, v_barriers, q_barriers = buffer_barriers
        k_consumed, v_consumed = consumed_barriers
        (chunk_row_indices_smem, chunk_tokens_smem,
         chunk_flat_positions_smem) = meta_smem
        compute_meta_barrier, memory_meta_barrier = meta_barriers
        
        # Grid indices
        batch_tile_idx = lax.axis_index("batch_tiles")
        head_idx = lax.axis_index("heads")
        wg_idx = lax.axis_index("wg")
        
        # KV head for GQA
        kv_head_idx = lax.div(head_idx, jnp.array(q_heads_per_kv_head, head_idx.dtype))
        
        # Base batch index for this tile
        batch_base = batch_tile_idx * block_q
        
        def perform_schedule_barrier():
            """Coordinate TensorCore usage between compute warpgroups."""
            if use_schedule_barrier:
                plgpu.barrier_arrive(schedule_barrier)
                plgpu.barrier_wait(schedule_barrier)
        
        # ---------------------------------------------------------------------
        # Compute Warpgroups (wg_idx < num_compute_wgs)
        # ---------------------------------------------------------------------
        @pl.when(wg_idx < num_compute_wgs)
        def _compute_wg():
            # Increase register budget for compute warpgroups
            plgpu.set_max_registers(232, action="increase")

            my_block_q = rows_per_wg
            row_start_within_tile = wg_idx * my_block_q
            q_slice_start = batch_base + row_start_within_tile

            # TMA copy this WG's Q rows to SMEM.
            plgpu.copy_gmem_to_smem(
                q_ref.at[pl.ds(q_slice_start, my_block_q), head_idx, :],
                qo_smem.at[wg_idx],
                q_barriers.at[wg_idx],
            )
            # TMA preload compute metadata (row_indices, tokens) to SMEM.
            # Issued while Q TMA is in flight for overlap.
            # When chunks > 256, TMA can't handle it — skip and read from GMEM.
            if use_smem_metadata:
                @pl.when(wg_idx == 0)
                def _preload_compute_metadata():
                    plgpu.copy_gmem_to_smem(
                        tile_chunk_row_indices_ref.at[batch_tile_idx],
                        chunk_row_indices_smem,
                        compute_meta_barrier,
                    )
                    plgpu.copy_gmem_to_smem(
                        tile_chunk_tokens_ref.at[batch_tile_idx],
                        chunk_tokens_smem,
                        compute_meta_barrier,
                    )
            plgpu.barrier_wait(q_barriers.at[wg_idx])
            if use_smem_metadata:
                plgpu.barrier_wait(compute_meta_barrier)

            # Initialize online softmax state (FlashAttention3)
            # m_i: running max (in log2 space for FMA)
            # l_i: running sum of exp(x - m)
            # acc: weighted sum of values
            m_i = plgpu.layout_cast(
                jnp.full((my_block_q,), -jnp.inf, dtype=jnp.float32),
                _WGMMA_ROW,
            )
            l_i = plgpu.layout_cast(
                jnp.full((my_block_q,), 0.0, dtype=jnp.float32),
                _WGMMA_ROW,
            )
            acc = plgpu.layout_cast(
                jnp.full((my_block_q, head_dim), 0.0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            num_kv_steps = tile_chunk_counts_ref[batch_tile_idx]

            # Create 2D row/col indices in WGMMA layout. This avoids 1D iota / WG_STRIDED
            # vector constraints (1D vectors must be multiples of 128 elements).
            row_ids_2d = plgpu.layout_cast(
                plgpu.broadcasted_iota(
                    jnp.int32, (my_block_q, block_kv), 0, layout=plgpu.Layout.WGMMA
                ),
                plgpu.Layout.WGMMA,
            )
            col_ids_2d = plgpu.layout_cast(
                plgpu.broadcasted_iota(
                    jnp.int32, (my_block_q, block_kv), 1, layout=plgpu.Layout.WGMMA
                ),
                plgpu.Layout.WGMMA,
            )
            log2e = math.log2(math.e)
            
            # -----------------------------------------------------------------
            # KV loop (FlashAttention3 online softmax) with lax.fori_loop.
            #
            # Iterates over ALL chunk steps across every row in this tile.
            # Each step processes one row's KV chunk; other rows are masked
            # to -inf. The NaN-safe softmax guard prevents -inf - (-inf).
            # -----------------------------------------------------------------
            def _kv_loop_body(kv_step, carry):
                acc, m_i, l_i = carry
                slot = lax.rem(kv_step, jnp.int32(max_concurrent_steps))

                # Read per-step chunk metadata from SMEM (fast) or GMEM (fallback).
                # chunk_row is tile-global; map it to this WG's local row range.
                if use_smem_metadata:
                    chunk_row = chunk_row_indices_smem[kv_step]
                    chunk_tokens = chunk_tokens_smem[kv_step]
                else:
                    chunk_row = tile_chunk_row_indices_ref[batch_tile_idx, kv_step]
                    chunk_tokens = tile_chunk_tokens_ref[batch_tile_idx, kv_step]
                local_chunk_row = chunk_row - row_start_within_tile

                def _compute_chunk(
                    state,
                    local_chunk_row=local_chunk_row,
                    chunk_tokens=chunk_tokens,
                ):
                    acc, m_i, l_i = state

                    # Wait for K tile from memory warpgroup, then compute QK^T.
                    plgpu.barrier_wait(k_barriers.at[slot])

                    def compute_qk(acc_ref):
                        plgpu.wgmma(
                            acc_ref,
                            qo_smem.at[wg_idx],
                            plgpu.transpose_ref(k_smem.at[slot], (1, 0)),
                        )
                        plgpu.wgmma_wait(0)
                        perform_schedule_barrier()
                        return acc_ref[...]

                    qk = pl.run_scoped(
                        compute_qk,
                        plgpu.ACC((my_block_q, block_kv), jnp.float32),
                    )

                    # Signal that K has been consumed
                    plgpu.barrier_arrive(k_consumed.at[slot])

                    # Apply scale
                    qk = qk * scale

                    # Only the chunk's owning row participates in this step; other rows
                    # are masked to -inf so they do not update the online softmax state.
                    mask_row = row_ids_2d == local_chunk_row
                    col_mask = col_ids_2d < chunk_tokens
                    mask = mask_row & col_mask
                    qk = jnp.where(mask, qk, -jnp.inf)

                    # ----- Online Softmax (log2/exp2 for FMA) -----
                    qk_max = qk.max(axis=1) * log2e
                    m_candidate = jnp.maximum(m_i, qk_max)
                    m_ij = m_candidate

                    # Guard: -inf - (-inf) = NaN → replace with 0.0
                    safe_diff = jnp.where(m_i == m_ij, 0.0, m_i - m_ij)
                    alpha = jnp.exp2(safe_diff)
                    m_i = m_ij

                    # Softmax weights (guard similarly for p)
                    p_exponent = qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0])
                    p = jnp.exp2(jnp.where(mask, p_exponent, -jnp.inf))
                    p = jnp.where(mask, p, 0.0)

                    # Update accumulator with rescaling
                    acc = acc * lax.broadcast_in_dim(alpha, acc.shape, [0])
                    l_i = l_i * alpha

                    p16 = p.astype(q.dtype)
                    p_sum_before_barriers = throughput_mode and (head_dim <= 128)
                    if p_sum_before_barriers:
                        l_i = l_i + p.sum(axis=1)
                        acc, p16 = lax.optimization_barrier((acc, p16))
                        l_i, m_i = lax.optimization_barrier((l_i, m_i))

                    # Barrier coordination before V access
                    perform_schedule_barrier()
                    plgpu.barrier_wait(v_barriers.at[slot])
                    if not p_sum_before_barriers:
                        l_i = l_i + p.sum(axis=1)

                    # PV Matmul: pv = P @ V, then acc += pv
                    # Use run_scoped (fresh ACC → 0-init) to avoid ACC.init row
                    # corruption observed at the 32-row boundary.
                    def compute_pv(pv_acc_ref):
                        plgpu.wgmma(pv_acc_ref, p16, v_smem.at[slot])
                        plgpu.wgmma_wait(0)
                        return pv_acc_ref[...]

                    pv = pl.run_scoped(
                        compute_pv,
                        plgpu.ACC((my_block_q, head_dim), jnp.float32),
                    )
                    acc = acc + pv

                    plgpu.barrier_arrive(v_consumed.at[slot])
                    return acc, m_i, l_i

                chunk_in_wg = (
                    (chunk_row >= row_start_within_tile)
                    & (chunk_row < (row_start_within_tile + my_block_q))
                )

                def _skip_chunk(state):
                    # Keep barrier ordering intact so the memory WG can safely
                    # recycle pipeline slots, but skip TensorCore math for chunks
                    # owned by the other compute WG.
                    plgpu.barrier_wait(k_barriers.at[slot])
                    # Match compute-path TensorCore scheduling points to avoid
                    # deadlocks when schedule barriers are enabled.
                    perform_schedule_barrier()
                    plgpu.barrier_arrive(k_consumed.at[slot])
                    perform_schedule_barrier()
                    plgpu.barrier_wait(v_barriers.at[slot])
                    plgpu.barrier_arrive(v_consumed.at[slot])
                    return state

                return lax.cond(
                    chunk_in_wg,
                    _compute_chunk,
                    _skip_chunk,
                    (acc, m_i, l_i),
                )

            acc, m_i, l_i = lax.fori_loop(
                0, num_kv_steps.astype(jnp.int32), _kv_loop_body,
                (acc, m_i, l_i),
            )

            # Coordinate before epilogue
            perform_schedule_barrier()

            if return_partials:
                acc_out_ref, l_out_ref, m_out_ref = out_refs
                acc_out_ref[pl.ds(q_slice_start, my_block_q), head_idx, :] = acc.astype(jnp.float32)
                l_out_ref[pl.ds(q_slice_start, my_block_q), head_idx] = l_i.astype(jnp.float32)
                m_out_ref[pl.ds(q_slice_start, my_block_q), head_idx] = m_i.astype(jnp.float32)
            else:
                # ----- Normalize Output -----
                # O = acc / l_i
                (out_ref,) = out_refs
                safe_l = jnp.maximum(l_i, 1e-9)
                acc = acc / lax.broadcast_in_dim(safe_l, (my_block_q, head_dim), [0])

                # Store to SMEM, then TMA to GMEM
                qo_smem.at[wg_idx][...] = acc.astype(q.dtype)
                plgpu.commit_smem()

                plgpu.copy_smem_to_gmem(
                    qo_smem.at[wg_idx],
                    out_ref.at[pl.ds(q_slice_start, my_block_q), head_idx, :],
                )
                plgpu.wait_smem_to_gmem(0)
        
        # ---------------------------------------------------------------------
        # Memory Warpgroup (wg_idx == num_compute_wgs)
        # ---------------------------------------------------------------------
        @pl.when(wg_idx == num_compute_wgs)
        def _memory_wg():
            # Reduce register budget for memory warpgroup
            plgpu.set_max_registers(40, action="decrease")

            # TMA preload memory metadata (block_indices, offsets) to SMEM.
            # When chunks > 256, skip TMA and read from GMEM directly.
            if use_smem_metadata:
                plgpu.copy_gmem_to_smem(
                    tile_chunk_flat_positions_ref.at[batch_tile_idx],
                    chunk_flat_positions_smem,
                    memory_meta_barrier,
                )
                plgpu.barrier_wait(memory_meta_barrier)

            chunk_count = tile_chunk_counts_ref[batch_tile_idx]

            def issue_chunk(chunk_idx, slot):
                """Issue TMA copies for a chunk (metadata from SMEM or GMEM)."""
                if use_smem_metadata:
                    chunk_pos = chunk_flat_positions_smem[chunk_idx]
                else:
                    chunk_pos = tile_chunk_flat_positions_ref[batch_tile_idx, chunk_idx]
                plgpu.copy_gmem_to_smem(
                    k_cache_flat_ref.at[pl.ds(chunk_pos, block_kv), kv_head_idx, :],
                    k_smem.at[slot],
                    k_barriers.at[slot],
                )
                plgpu.copy_gmem_to_smem(
                    v_cache_flat_ref.at[pl.ds(chunk_pos, block_kv), kv_head_idx, :],
                    v_smem.at[slot],
                    v_barriers.at[slot],
                )

            # Prefill initial pipeline slots (static unroll, small count)
            for i in range(max_concurrent_steps):
                idx = jnp.array(i, dtype=jnp.int32)

                @pl.when(chunk_count > idx)
                def _prefill_slot(chunk_idx=i, slot=i):
                    issue_chunk(jnp.int32(chunk_idx), jnp.int32(slot))

            # Stream remaining chunks using pl.loop (dynamic)
            @pl.loop(0, chunk_count - max_concurrent_steps)
            def _stream_loop(step):
                chunk_idx = step + max_concurrent_steps
                slot = lax.rem(step, jnp.int32(max_concurrent_steps))
                plgpu.barrier_wait(k_consumed.at[slot])
                plgpu.barrier_wait(v_consumed.at[slot])
                issue_chunk(chunk_idx, slot)
    
    # Launch kernel
    k_cache_flat = k_cache.reshape(num_kv_blocks * kv_block_size, num_kv_heads, head_dim)
    v_cache_flat = v_cache.reshape(num_kv_blocks * kv_block_size, num_kv_heads, head_dim)

    kernel_args = (
        q,
        k_cache_flat,
        v_cache_flat,
        tile_valid_counts,
        tile_row_offsets,
        tile_row_lengths,
        tile_chunk_block_indices,
        tile_chunk_offsets,
        tile_chunk_flat_positions,
        tile_chunk_tokens,
        tile_chunk_prefix_tokens,
        tile_chunk_row_indices,
        tile_chunk_logical_blocks,
        tile_chunk_counts,
    )
    compiler_params = _decode_compiler_params(throughput_mode=throughput_mode)

    if return_partials:
        acc_partial, l_partial, m_partial = plgpu.kernel(
            kernel_entry_partials,
            out_shape=[
                jax.ShapeDtypeStruct(q.shape, jnp.float32),
                jax.ShapeDtypeStruct((q.shape[0], q.shape[1]), jnp.float32),
                jax.ShapeDtypeStruct((q.shape[0], q.shape[1]), jnp.float32),
            ],
            grid=(num_batch_tiles, num_heads),
            grid_names=("batch_tiles", "heads"),
            num_threads=num_compute_wgs + 1,  # 2 compute + 1 memory
            thread_name="wg",
            compiler_params=compiler_params,
        )(*kernel_args)
        return (
            acc_partial[:original_batch_size],
            l_partial[:original_batch_size],
            m_partial[:original_batch_size],
        )

    out = plgpu.kernel(
        kernel_entry,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(num_batch_tiles, num_heads),
        grid_names=("batch_tiles", "heads"),
        num_threads=num_compute_wgs + 1,  # 2 compute + 1 memory
        thread_name="wg",
        compiler_params=compiler_params,
    )(*kernel_args)

    return out[:original_batch_size]


# =============================================================================
# Decode Rewrite Kernel (Algorithmic Pivot: Partitioned Streaming)
# =============================================================================

def _merge_split_partials_streaming(
    acc_partial: jax.Array,
    l_partial: jax.Array,
    m_partial: jax.Array,
    *,
    axis: int,
    eps: float = 1e-9,
) -> jax.Array:
    """Merge split partial outputs via streaming online-softmax recurrence.

    This avoids materializing a full correction tensor for all splits at once,
    reducing temporary memory pressure in latency/throughput reduction paths.
    """

    # Bring split axis to position 1 to keep the scan body simple.
    if axis != 1:
        acc_partial = jnp.moveaxis(acc_partial, axis, 1)
        l_partial = jnp.moveaxis(l_partial, axis, 1)
        m_partial = jnp.moveaxis(m_partial, axis, 1)

    # Fast path: a single split needs only normalization.
    if acc_partial.shape[1] == 1:
        acc0 = acc_partial[:, 0, :, :].astype(jnp.float32)
        l0 = l_partial[:, 0, :].astype(jnp.float32)
        valid0 = l0 > 0
        out0 = acc0 / jnp.maximum(l0[..., None], eps)
        return jnp.where(valid0[..., None], out0, 0.0)

    # Initialize recurrence from split 0 to avoid one scan step and redundant
    # float32 casts inside the body.
    init_acc = acc_partial[:, 0, :, :].astype(jnp.float32)
    init_l = l_partial[:, 0, :].astype(jnp.float32)
    init_m = m_partial[:, 0, :].astype(jnp.float32)
    init_valid = init_l > 0
    init_acc = init_acc * init_valid[..., None].astype(jnp.float32)
    init_m = jnp.where(init_valid, init_m, -jnp.inf)

    acc_final = init_acc
    l_final = init_l
    m_final = init_m
    # Split count is static in compiled decode paths; unroll to avoid a
    # runtime while/scan in lowered IR.
    for split_idx in range(1, int(acc_partial.shape[1])):
        acc_s = acc_partial[:, split_idx, :, :].astype(jnp.float32)
        l_s = l_partial[:, split_idx, :].astype(jnp.float32)
        m_s = m_partial[:, split_idx, :].astype(jnp.float32)
        final_valid = l_final > 0
        split_valid = l_s > 0
        m_final_safe = jnp.where(final_valid, m_final, -jnp.inf)
        m_s_safe = jnp.where(split_valid, m_s, -jnp.inf)
        m_next = jnp.maximum(m_final_safe, m_s_safe)
        alpha = jnp.where(final_valid, jnp.exp2(m_final_safe - m_next), 0.0)
        beta = jnp.where(split_valid, jnp.exp2(m_s_safe - m_next), 0.0)
        acc_final = acc_final * alpha[..., None] + acc_s * beta[..., None]
        l_final = l_final * alpha + l_s * beta
        m_final = jnp.where(final_valid | split_valid, m_next, m_final)

    return acc_final / jnp.maximum(l_final[..., None], eps)


def _merge_split_partials(
    acc_partial: jax.Array,
    l_partial: jax.Array,
    m_partial: jax.Array,
    *,
    axis: int,
    eps: float = 1e-9,
) -> jax.Array:
    """Compatibility wrapper for the current streaming reduction implementation."""
    return _merge_split_partials_streaming(
        acc_partial,
        l_partial,
        m_partial,
        axis=axis,
        eps=eps,
    )


@partial(jax.jit, static_argnames=("axis", "eps"))
def _merge_split_partials_device(
    acc_partial: jax.Array,
    l_partial: jax.Array,
    m_partial: jax.Array,
    *,
    axis: int,
    eps: float = 1e-9,
) -> jax.Array:
    """Compile the streaming recurrence into a device-side reduction boundary."""
    return _merge_split_partials_streaming(
        acc_partial,
        l_partial,
        m_partial,
        axis=axis,
        eps=eps,
    )


def reduce_partitioned_decode_partials(
    acc_partial: jax.Array,
    l_partial: jax.Array,
    m_partial: jax.Array,
    *,
    axis: int,
    family: str,
    backend_override: str | None = None,
) -> jax.Array:
    """Internal reduction boundary for partitioned decode families."""
    backend = _active_partitioned_decode_reduction_backend(backend_override)
    profiling_enabled = decode_step_profiling_enabled()
    can_measure = profiling_enabled and not _tree_has_tracer(
        (acc_partial, l_partial, m_partial)
    )
    started_at = perf_counter() if can_measure else 0.0

    if backend == "streaming":
        out = _merge_split_partials_streaming(
            acc_partial,
            l_partial,
            m_partial,
            axis=axis,
        )
    elif backend == "device":
        out = _merge_split_partials_device(
            acc_partial,
            l_partial,
            m_partial,
            axis=axis,
        )
    else:  # pragma: no cover - backend normalization guards this.
        raise ValueError(f"Unsupported partitioned decode reduction backend: {backend}")

    if profiling_enabled:
        if can_measure:
            block_until_ready_tree(out)
        record_partitioned_decode_reduction_stats(
            seconds=(perf_counter() - started_at) if can_measure else 0.0,
            backend=backend,
            family=family,
            splits=int(acc_partial.shape[axis]),
            measured=can_measure,
        )
    return out


def _select_throughput_k_splits(
    *,
    split_k: int,
    batch_size: int,
    head_dim: int,
    num_heads: int,
    block_q: int,
    max_blocks_per_seq: int,
    block_size: int,
    block_kv: int,
) -> int:
    """Pick split count for the throughput decode path with optional table override."""

    if max_blocks_per_seq <= 1:
        return 1
    chunks_per_block = block_size // block_kv
    num_kv_tiles = max_blocks_per_seq * chunks_per_block
    if num_kv_tiles <= 1:
        return 1

    override = _lookup_throughput_splitk_override(
        batch_size=batch_size,
        head_dim=head_dim,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
    )
    if override is not None and split_k <= 0:
        target = min(int(override), max_blocks_per_seq, num_kv_tiles)
        while target > 1 and (num_kv_tiles % target):
            target -= 1
        return max(1, target)

    if split_k > 0:
        target = min(int(split_k), max_blocks_per_seq, num_kv_tiles)
    else:
        # Empirical H100 defaults:
        # - Split-k helps for medium/long contexts (16-32 blocks).
        # - Split-k often regresses for very long contexts (>=48 blocks) due
        #   extra partial-merge traffic.
        if max_blocks_per_seq >= 48:
            target = 1
        elif max_blocks_per_seq >= 32 and batch_size >= 512:
            target = 8
        elif max_blocks_per_seq >= 16 and batch_size >= 256:
            target = 4
        else:
            base_tiles = math.ceil(batch_size / block_q) * num_heads
            num_ctas = max(1, _default_core_count())
            load_factor = base_tiles / num_ctas
            if load_factor >= 0.5:
                return 1
            target = max(2, num_ctas // max(base_tiles, 1))
            target = min(target, max_blocks_per_seq, num_kv_tiles, 8)

    while target > 1 and (num_kv_tiles % target):
        target -= 1
    return max(1, target)


def _select_latency_k_splits(max_blocks_per_seq: int, head_dim: int) -> int:
    """Pick partition count for the latency-focused decode kernel."""
    if max_blocks_per_seq >= 32:
        return 8 if head_dim >= 128 else 4
    if max_blocks_per_seq >= 16:
        return 4
    if max_blocks_per_seq >= 8:
        return 2
    return 1


def _resolve_latency_config(block_size: int, config: MosaicAttentionConfig | None) -> MosaicAttentionConfig:
    """Normalize latency-path tuning so block_kv is valid for the KV page size."""
    if config is None:
        block_kv = block_size if block_size % 64 == 0 else 64
        block_kv = min(block_kv, 256)
        return MosaicAttentionConfig(
            block_q=64,
            block_kv=block_kv,
            max_concurrent_steps=2,
            use_schedule_barrier=True,
            num_compute_wgs=2,
        )

    block_kv = min(config.block_kv, block_size)
    block_kv = max(64, (block_kv // 64) * 64)
    while block_kv > 64 and (block_size % block_kv != 0):
        block_kv -= 64
    if block_size % block_kv != 0:
        block_kv = 64
    if block_size % block_kv != 0:
        raise ValueError(
            f"latency config requires block_size divisible by block_kv: "
            f"block_size={block_size}, resolved_block_kv={block_kv}"
        )
    return MosaicAttentionConfig(
        block_q=config.block_q,
        block_kv=block_kv,
        max_concurrent_steps=config.max_concurrent_steps,
        use_schedule_barrier=config.use_schedule_barrier,
        num_compute_wgs=config.num_compute_wgs,
    )


def _resolve_throughput_v2_config(
    block_size: int,
    config: MosaicAttentionConfig | None,
) -> MosaicAttentionConfig:
    """Normalize throughput-v2 to the expert-guided v1 defaults.

    Throughput-v2 is row-native and does not depend on the old batched decode
    core's multi-WG + schedule-barrier defaults. The current v1 path only
    supports the expert-guided execution envelope:
    block_q=64, 1 compute WG, 2 stages, no schedule barrier.
    Caller-provided configs may still tighten block_kv, but the unsupported
    launch topology fields are normalized to the current implementation.
    """
    if config is None:
        return MosaicAttentionConfig(
            block_q=64,
            block_kv=64 if block_size % 64 == 0 else block_size,
            max_concurrent_steps=2,
            use_schedule_barrier=False,
            num_compute_wgs=1,
        )

    block_kv = min(config.block_kv, block_size)
    block_kv = max(64, (block_kv // 64) * 64)
    while block_kv > 64 and (block_size % block_kv != 0):
        block_kv -= 64
    if block_size % block_kv != 0:
        block_kv = 64
    if block_size % block_kv != 0:
        raise ValueError(
            "throughput_v2 config requires block_size divisible by block_kv: "
            f"block_size={block_size}, resolved_block_kv={block_kv}"
        )
    return MosaicAttentionConfig(
        block_q=64,
        block_kv=block_kv,
        max_concurrent_steps=2,
        use_schedule_barrier=False,
        num_compute_wgs=1,
    )


def _should_use_throughput_v2_mosaic_kernel() -> bool:
    return (
        MOSAIC_AVAILABLE
        and jax.default_backend() == "gpu"
        and os.environ.get("NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC", "").strip() == "1"
    )


def paged_decode_attention_mosaic_latency(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    block_size: int,
    config: MosaicAttentionConfig | None = None,
    prepared_metadata_cache: dict[tuple, object] | None = None,
) -> jax.Array:
    """Partitioned decode path tuned for short-context latency.

    The initial partitioned implementation used one-program-per-sequence partition loops,
    which underutilized H100 TensorCores (M dimension too small). This version
    runs each split through the high-throughput batched Mosaic decode core and
    merges split residuals with online-softmax correction terms.
    """
    _check_mosaic_available()

    batch_size, num_heads, head_dim = q.shape
    _, cache_block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if cache_block_size != block_size:
        raise ValueError(f"block_size mismatch: cache={cache_block_size} arg={block_size}")
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim} cache={cache_head_dim}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    max_blocks_per_seq = block_tables.shape[1]
    if max_blocks_per_seq == 0:
        return jnp.zeros_like(q)
    kernel_config = _resolve_latency_config(block_size, config)

    # The latency path is beneficial primarily for wider heads.
    # For narrow heads, keep the stable baseline core.
    if head_dim < 128:
        return batched_decode_attention_mosaic(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            config=kernel_config,
        )

    k_splits = min(
        max_blocks_per_seq,
        _select_latency_k_splits(max_blocks_per_seq, head_dim),
    )
    pages_per_partition = (max_blocks_per_seq + k_splits - 1) // k_splits
    if pages_per_partition <= 0:
        return jnp.zeros_like(q)
    if k_splits <= 1:
        return batched_decode_attention_mosaic(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            config=kernel_config,
        )

    padded_blocks = pages_per_partition * k_splits
    pad_cols = padded_blocks - max_blocks_per_seq
    if pad_cols:
        block_tables = jnp.pad(block_tables, ((0, 0), (0, pad_cols)), constant_values=0)
    split_block_tables = block_tables.reshape(batch_size, k_splits, pages_per_partition)
    split_base_tokens = (
        jnp.arange(k_splits, dtype=jnp.int32) * (pages_per_partition * block_size)
    )
    split_context_lens = jnp.clip(
        context_lens[:, None] - split_base_tokens[None, :],
        min=0,
        max=pages_per_partition * block_size,
    ).astype(jnp.int32)

    q_split = jnp.broadcast_to(
        q[:, None, :, :],
        (batch_size, k_splits, num_heads, head_dim),
    ).reshape(batch_size * k_splits, num_heads, head_dim)
    block_tables_split = split_block_tables.reshape(batch_size * k_splits, pages_per_partition)
    context_lens_split = split_context_lens.reshape(batch_size * k_splits)

    cache_key = (
        "latency",
        batch_size,
        k_splits,
        pages_per_partition,
        kernel_config.block_q,
        kernel_config.block_kv,
        block_size,
    )
    metadata = _get_or_prepare_cached_metadata(
        prepared_metadata_cache,
        cache_key,
        lambda: prepare_decode_metadata(
            block_tables_split,
            context_lens_split,
            q_split.shape[0],
            kernel_config.block_q,
            block_size,
            kernel_config.block_kv,
            include_unused_fields=False,
        ),
    )

    acc_partial, l_partial, m_partial = batched_decode_attention_mosaic(
        q=q_split,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables_split,
        context_lens=context_lens_split,
        scale=scale,
        config=kernel_config,
        metadata=metadata,
        return_partials=True,
    )

    acc_partial = acc_partial.reshape(batch_size, k_splits, num_heads, head_dim)
    l_partial = l_partial.reshape(batch_size, k_splits, num_heads)
    m_partial = m_partial.reshape(batch_size, k_splits, num_heads)

    out = reduce_partitioned_decode_partials(
        acc_partial,
        l_partial,
        m_partial,
        axis=1,
        family="latency",
    )
    return out.astype(q.dtype)


# =============================================================================
# Decode Throughput Kernel (Split-K + Manual-Barrier Scheduling)
# =============================================================================

def paged_decode_attention_mosaic_throughput(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    block_size: int,
    config: MosaicAttentionConfig | None = None,
    split_k: int = 0,
    rescale_threshold: float = 1.0,
    autotune: bool = False,
    prepared_metadata_cache: dict[tuple, object] | None = None,
) -> jax.Array:
    """Long-context throughput path with split-k and manual-barrier tuning."""
    _check_mosaic_available()
    del autotune  # Optional knob reserved for future local tuning sweeps.

    batch_size, num_heads, head_dim = q.shape
    _, cache_block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if cache_block_size != block_size:
        raise ValueError(f"block_size mismatch: cache={cache_block_size} arg={block_size}")
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim} cache={cache_head_dim}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if rescale_threshold != 1.0:
        raise ValueError(
            "throughput decode currently supports rescale_threshold=1.0 only"
        )

    max_blocks_per_seq = block_tables.shape[1]
    if max_blocks_per_seq == 0:
        return jnp.zeros_like(q)

    kernel_config = _resolve_latency_config(block_size, config)
    if head_dim < 128 or max_blocks_per_seq < 16:
        return batched_decode_attention_mosaic(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            config=kernel_config,
            throughput_mode=True,
        )

    k_splits = _select_throughput_k_splits(
        split_k=split_k,
        batch_size=batch_size,
        head_dim=head_dim,
        num_heads=num_heads,
        block_q=kernel_config.block_q,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        block_kv=kernel_config.block_kv,
    )
    pages_per_partition = (max_blocks_per_seq + k_splits - 1) // k_splits
    if pages_per_partition <= 0:
        return jnp.zeros_like(q)

    # Fast path: no split, reuse one metadata schedule across layers.
    if k_splits <= 1:
        cache_key = (
            "throughput",
            batch_size,
            1,
            pages_per_partition,
            kernel_config.block_q,
            kernel_config.block_kv,
            block_size,
        )
        metadata = _get_or_prepare_cached_metadata(
            prepared_metadata_cache,
            cache_key,
            lambda: prepare_decode_metadata(
                block_tables,
                context_lens,
                q.shape[0],
                kernel_config.block_q,
                block_size,
                kernel_config.block_kv,
                include_unused_fields=False,
            ),
        )
        return batched_decode_attention_mosaic(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            config=kernel_config,
            metadata=metadata,
            throughput_mode=True,
        )

    padded_blocks = pages_per_partition * k_splits
    pad_cols = padded_blocks - max_blocks_per_seq
    if pad_cols:
        block_tables = jnp.pad(block_tables, ((0, 0), (0, pad_cols)), constant_values=0)
    split_block_tables = block_tables.reshape(batch_size, k_splits, pages_per_partition)
    split_base_tokens = (
        jnp.arange(k_splits, dtype=jnp.int32) * (pages_per_partition * block_size)
    )
    split_context_lens = jnp.clip(
        context_lens[:, None] - split_base_tokens[None, :],
        min=0,
        max=pages_per_partition * block_size,
    ).astype(jnp.int32)

    q_split = jnp.broadcast_to(
        q[:, None, :, :],
        (batch_size, k_splits, num_heads, head_dim),
    ).reshape(batch_size * k_splits, num_heads, head_dim)
    block_tables_split = split_block_tables.reshape(batch_size * k_splits, pages_per_partition)
    context_lens_split = split_context_lens.reshape(batch_size * k_splits)

    cache_key = (
        "throughput",
        batch_size,
        k_splits,
        pages_per_partition,
        kernel_config.block_q,
        kernel_config.block_kv,
        block_size,
    )
    metadata = _get_or_prepare_cached_metadata(
        prepared_metadata_cache,
        cache_key,
        lambda: prepare_decode_metadata(
            block_tables_split,
            context_lens_split,
            q_split.shape[0],
            kernel_config.block_q,
            block_size,
            kernel_config.block_kv,
            include_unused_fields=False,
        ),
    )

    acc_partial, l_partial, m_partial = batched_decode_attention_mosaic(
        q=q_split,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables_split,
        context_lens=context_lens_split,
        scale=scale,
        config=kernel_config,
        metadata=metadata,
        return_partials=True,
        throughput_mode=True,
    )

    acc_partial = acc_partial.reshape(batch_size, k_splits, num_heads, head_dim)
    l_partial = l_partial.reshape(batch_size, k_splits, num_heads)
    m_partial = m_partial.reshape(batch_size, k_splits, num_heads)
    out = reduce_partitioned_decode_partials(
        acc_partial,
        l_partial,
        m_partial,
        axis=1,
        family="throughput",
    )
    return out.astype(q.dtype)


def build_paged_decode_throughput_v2_plan(
    *,
    q: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    block_size: int,
    num_kv_heads: int | None = None,
    config: MosaicAttentionConfig | None = None,
    split_k: int = 0,
) -> ThroughputV2Plan:
    """Build the tiny schedule-owned plan for the bridge-free throughput_v2 path."""
    del context_lens
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = int(num_kv_heads or num_heads)
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    max_blocks_per_seq = block_tables.shape[1]
    kernel_config = _resolve_throughput_v2_config(block_size, config)
    k_splits = _select_throughput_k_splits(
        split_k=split_k,
        batch_size=batch_size,
        head_dim=head_dim,
        num_heads=num_heads,
        block_q=kernel_config.block_q,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        block_kv=kernel_config.block_kv,
    )
    pages_per_partition = (
        (max_blocks_per_seq + k_splits - 1) // k_splits if k_splits > 0 else 0
    )
    q_heads_per_kv_head = num_heads // num_kv_heads
    if _should_use_throughput_v2_mosaic_kernel():
        partial_kernel = "row_partition_mosaic_v1"
        launch_block_q = 64
        launch_max_concurrent_steps = 2
        launch_num_compute_wgs = 1
        launch_num_memory_wgs = 1
        launch_use_schedule_barrier = False
    else:
        partial_kernel = "row_partition_jax_v1"
        launch_block_q = None
        launch_max_concurrent_steps = None
        launch_num_compute_wgs = None
        launch_num_memory_wgs = None
        launch_use_schedule_barrier = None
    return ThroughputV2Plan(
        batch_size=batch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        block_kv=kernel_config.block_kv,
        q_heads_per_kv_head=q_heads_per_kv_head,
        k_splits=k_splits,
        pages_per_partition=pages_per_partition,
        max_concurrent_steps=kernel_config.max_concurrent_steps,
        num_compute_wgs=kernel_config.num_compute_wgs,
        use_schedule_barrier=kernel_config.use_schedule_barrier,
        partial_kernel=partial_kernel,
        launch_block_q=launch_block_q,
        launch_max_concurrent_steps=launch_max_concurrent_steps,
        launch_num_compute_wgs=launch_num_compute_wgs,
        launch_num_memory_wgs=launch_num_memory_wgs,
        launch_use_schedule_barrier=launch_use_schedule_barrier,
        uses_wrapper_partitioning=False,
        uses_batched_core=False,
        reduction_boundary="device_split_reduction_v1",
        reduction_backend="device",
        metadata_model="schedule_plan_v1",
        metadata_cache_key=(
            "throughput_v2",
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_blocks_per_seq,
            block_size,
            kernel_config.block_kv,
            q_heads_per_kv_head,
            k_splits,
            pages_per_partition,
            kernel_config.max_concurrent_steps,
            kernel_config.num_compute_wgs,
            kernel_config.use_schedule_barrier,
            partial_kernel,
            "device_split_reduction_v1",
            "schedule_plan_v1",
        ),
    )


def _throughput_v2_compute_partition_partials(
    *,
    q_grouped: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    plan: ThroughputV2Plan,
    split_idx: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute row-native partials for one `(row, kv_head_group, partition)` view."""
    batch_size, num_kv_heads, q_heads_per_kv_head, head_dim = q_grouped.shape
    num_blocks = k_cache.shape[0]
    chunks_per_block = plan.block_size // plan.block_kv
    start_page = split_idx * plan.pages_per_partition
    start_token = start_page * plan.block_size
    max_partition_tokens = plan.pages_per_partition * plan.block_size
    split_context_lens = jnp.clip(
        context_lens.astype(jnp.int32) - jnp.int32(start_token),
        min=0,
        max=max_partition_tokens,
    ).astype(jnp.int32)

    acc = jnp.zeros(
        (batch_size, num_kv_heads, q_heads_per_kv_head, head_dim),
        dtype=jnp.float32,
    )
    l = jnp.zeros(
        (batch_size, num_kv_heads, q_heads_per_kv_head),
        dtype=jnp.float32,
    )
    m = jnp.full_like(l, -jnp.inf)
    log2e = jnp.float32(math.log2(math.e))
    token_positions = jnp.arange(plan.block_kv, dtype=jnp.int32)[None, None, None, :]

    for local_page_idx in range(plan.pages_per_partition):
        global_page_idx = start_page + local_page_idx
        if global_page_idx >= plan.max_blocks_per_seq:
            break

        phys = jnp.clip(block_tables[:, global_page_idx], min=0, max=num_blocks - 1)
        k_page = k_cache[phys]
        v_page = v_cache[phys]
        k_page = jnp.transpose(k_page, (0, 2, 1, 3)).astype(jnp.float32)
        v_page = jnp.transpose(v_page, (0, 2, 1, 3)).astype(jnp.float32)
        page_context_lens = jnp.clip(
            split_context_lens - jnp.int32(local_page_idx * plan.block_size),
            min=0,
            max=plan.block_size,
        ).astype(jnp.int32)

        for chunk_idx in range(chunks_per_block):
            chunk_start = chunk_idx * plan.block_kv
            valid_tokens = jnp.clip(
                page_context_lens - jnp.int32(chunk_start),
                min=0,
                max=plan.block_kv,
            ).astype(jnp.int32)
            k_chunk = lax.dynamic_slice_in_dim(
                k_page,
                start_index=chunk_start,
                slice_size=plan.block_kv,
                axis=2,
            )
            v_chunk = lax.dynamic_slice_in_dim(
                v_page,
                start_index=chunk_start,
                slice_size=plan.block_kv,
                axis=2,
            )
            logits = jnp.einsum(
                "bgqd,bgkd->bgqk",
                q_grouped,
                k_chunk,
                preferred_element_type=jnp.float32,
            ) * scale
            mask = token_positions < valid_tokens[:, None, None, None]
            logits = jnp.where(mask, logits, -jnp.inf)

            current_valid = l > 0
            chunk_valid = valid_tokens[:, None, None] > 0
            m_safe = jnp.where(current_valid, m, -jnp.inf)
            m_curr = jnp.max(logits, axis=-1)
            m_curr_safe = jnp.where(chunk_valid, m_curr, -jnp.inf)
            m_next = jnp.maximum(m_safe, m_curr_safe)
            any_valid = current_valid | chunk_valid
            corr = jnp.where(current_valid, jnp.exp2((m_safe - m_next) * log2e), 0.0)
            m_next_for_exp = jnp.where(any_valid, m_next, 0.0)
            logits_safe = jnp.where(mask, logits, m_next_for_exp[..., None])
            p = jnp.exp2((logits_safe - m_next_for_exp[..., None]) * log2e)
            p = jnp.where(mask, p, 0.0)
            l = l * corr + p.sum(axis=-1)
            acc = acc * corr[..., None] + jnp.einsum(
                "bgqk,bgkd->bgqd",
                p,
                v_chunk,
                preferred_element_type=jnp.float32,
            )
            m = jnp.where(any_valid, m_next, m)

    return acc, l, m


def _compute_throughput_v2_partials_jax(
    *,
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    plan: ThroughputV2Plan,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX fallback for throughput_v2 partials."""
    q_grouped = q.reshape(
        plan.batch_size,
        plan.num_kv_heads,
        plan.q_heads_per_kv_head,
        plan.head_dim,
    )
    acc_partials = []
    l_partials = []
    m_partials = []
    for split_idx in range(plan.k_splits):
        acc_s, l_s, m_s = _throughput_v2_compute_partition_partials(
            q_grouped=q_grouped,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            plan=plan,
            split_idx=split_idx,
        )
        acc_partials.append(acc_s)
        l_partials.append(l_s)
        m_partials.append(m_s)

    acc_grouped = jnp.stack(acc_partials, axis=1)
    l_grouped = jnp.stack(l_partials, axis=1)
    m_grouped = jnp.stack(m_partials, axis=1)
    acc_partial = acc_grouped.reshape(
        plan.batch_size,
        plan.k_splits,
        plan.num_heads,
        plan.head_dim,
    )
    l_partial = l_grouped.reshape(plan.batch_size, plan.k_splits, plan.num_heads)
    m_partial = m_grouped.reshape(plan.batch_size, plan.k_splits, plan.num_heads)
    return acc_partial, l_partial, m_partial


def _compute_throughput_v2_partials_mosaic(
    *,
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    plan: ThroughputV2Plan,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Row-native throughput_v2 partials kernel backed by Pallas Mosaic GPU."""
    _check_mosaic_available()
    padded_block_q = 64
    if plan.q_heads_per_kv_head > padded_block_q:
        raise NotImplementedError(
            "throughput_v2 mosaic partials currently support at most 64 query heads per KV head group"
        )

    q_grouped = q.reshape(
        plan.batch_size,
        plan.num_kv_heads,
        plan.q_heads_per_kv_head,
        plan.head_dim,
    )
    q_grouped_padded = jnp.pad(
        q_grouped,
        ((0, 0), (0, 0), (0, padded_block_q - plan.q_heads_per_kv_head), (0, 0)),
    )
    num_blocks = k_cache.shape[0]
    chunks_per_block = plan.block_size // plan.block_kv
    max_partition_tokens = plan.pages_per_partition * plan.block_size
    k_cache_flat = k_cache.reshape(num_blocks * plan.block_size, plan.num_kv_heads, plan.head_dim)
    v_cache_flat = v_cache.reshape(num_blocks * plan.block_size, plan.num_kv_heads, plan.head_dim)
    scale_arr = jnp.asarray(scale, dtype=jnp.float32)
    transforms = get_smem_transforms(plan.head_dim, q.dtype)
    compiler_params = _decode_compiler_params(throughput_mode=True)

    def kernel_entry(
        q_ref,
        k_cache_flat_ref,
        v_cache_flat_ref,
        block_tables_ref,
        context_lens_ref,
        scale_ref,
        acc_ref,
        l_ref,
        m_ref,
        q_smem,
        k_smem,
        v_smem,
        q_barrier,
        k_barrier,
        v_barrier,
        k_consumed,
        v_consumed,
    ):
        batch_idx = lax.axis_index("batch")
        kv_head_idx = lax.axis_index("kv_heads")
        split_idx = lax.axis_index("splits")
        wg_idx = lax.axis_index("wg")

        split_start_page = split_idx * plan.pages_per_partition
        split_start_token = split_start_page * plan.block_size
        split_context_len = jnp.clip(
            context_lens_ref[batch_idx].astype(jnp.int32) - jnp.int32(split_start_token),
            min=0,
            max=max_partition_tokens,
        ).astype(jnp.int32)

        @pl.when(wg_idx == 1)
        def _memory_wg():
            plgpu.set_max_registers(40, action="decrease")
            plgpu.copy_gmem_to_smem(
                q_ref.at[batch_idx, kv_head_idx],
                q_smem,
                q_barrier,
            )

            first_chunk = True
            for local_page_idx in range(plan.pages_per_partition):
                global_page_idx = split_start_page + local_page_idx
                safe_page_idx = jnp.minimum(
                    global_page_idx,
                    jnp.int32(plan.max_blocks_per_seq - 1),
                )
                physical_block = jnp.clip(
                    block_tables_ref[batch_idx, safe_page_idx],
                    min=0,
                    max=num_blocks - 1,
                )

                for chunk_idx in range(chunks_per_block):
                    if not first_chunk:
                        plgpu.barrier_wait(k_consumed)
                        plgpu.barrier_wait(v_consumed)
                    first_chunk = False
                    chunk_base = physical_block * plan.block_size + chunk_idx * plan.block_kv
                    plgpu.copy_gmem_to_smem(
                        k_cache_flat_ref.at[
                            pl.ds(chunk_base, plan.block_kv),
                            kv_head_idx,
                        ],
                        k_smem,
                        k_barrier,
                    )
                    plgpu.copy_gmem_to_smem(
                        v_cache_flat_ref.at[
                            pl.ds(chunk_base, plan.block_kv),
                            kv_head_idx,
                        ],
                        v_smem,
                        v_barrier,
                    )

        @pl.when(wg_idx == 0)
        def _compute_wg():
            plgpu.set_max_registers(232, action="increase")
            scale_value = scale_ref[...].astype(jnp.float32)
            plgpu.barrier_wait(q_barrier)

            m_i = plgpu.layout_cast(
                jnp.full((padded_block_q,), -jnp.inf, dtype=jnp.float32),
                _WGMMA_ROW,
            )
            l_i = plgpu.layout_cast(
                jnp.zeros((padded_block_q,), dtype=jnp.float32),
                _WGMMA_ROW,
            )
            acc = plgpu.layout_cast(
                jnp.zeros((padded_block_q, plan.head_dim), dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            row_ids_2d = plgpu.layout_cast(
                plgpu.broadcasted_iota(
                    jnp.int32,
                    (padded_block_q, plan.block_kv),
                    0,
                    layout=plgpu.Layout.WGMMA,
                ),
                plgpu.Layout.WGMMA,
            )
            col_ids_2d = plgpu.layout_cast(
                plgpu.broadcasted_iota(
                    jnp.int32,
                    (padded_block_q, plan.block_kv),
                    1,
                    layout=plgpu.Layout.WGMMA,
                ),
                plgpu.Layout.WGMMA,
            )
            log2e = math.log2(math.e)

            for local_page_idx in range(plan.pages_per_partition):
                page_context_len = jnp.clip(
                    split_context_len - jnp.int32(local_page_idx * plan.block_size),
                    min=0,
                    max=plan.block_size,
                ).astype(jnp.int32)

                for chunk_idx in range(chunks_per_block):
                    valid_tokens = jnp.clip(
                        page_context_len - jnp.int32(chunk_idx * plan.block_kv),
                        min=0,
                        max=plan.block_kv,
                    ).astype(jnp.int32)
                    plgpu.barrier_wait(k_barrier)

                    def compute_qk(acc_ref):
                        plgpu.wgmma(
                            acc_ref,
                            q_smem,
                            plgpu.transpose_ref(k_smem, (1, 0)),
                        )
                        plgpu.wgmma_wait(0)
                        return acc_ref[...]

                    qk = pl.run_scoped(
                        compute_qk,
                        plgpu.ACC((padded_block_q, plan.block_kv), jnp.float32),
                    )
                    plgpu.barrier_arrive(k_consumed)

                    qk = qk * scale_value
                    valid_row_mask = row_ids_2d < plan.q_heads_per_kv_head
                    col_mask = col_ids_2d < valid_tokens
                    mask = valid_row_mask & col_mask
                    qk = jnp.where(mask, qk, -jnp.inf)

                    qk_max = qk.max(axis=1) * log2e
                    m_candidate = jnp.maximum(m_i, qk_max)
                    m_ij = m_candidate
                    safe_diff = jnp.where(m_i == m_ij, 0.0, m_i - m_ij)
                    alpha = jnp.exp2(safe_diff)
                    m_i = m_ij
                    p_exponent = qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0])
                    p = jnp.exp2(jnp.where(mask, p_exponent, -jnp.inf))
                    p = jnp.where(mask, p, 0.0)
                    acc = acc * lax.broadcast_in_dim(alpha, acc.shape, [0])
                    l_i = l_i * alpha + p.sum(axis=1)

                    p16 = p.astype(q.dtype)
                    plgpu.barrier_wait(v_barrier)

                    def compute_pv(pv_acc_ref):
                        plgpu.wgmma(pv_acc_ref, p16, v_smem)
                        plgpu.wgmma_wait(0)
                        return pv_acc_ref[...]

                    pv = pl.run_scoped(
                        compute_pv,
                        plgpu.ACC((padded_block_q, plan.head_dim), jnp.float32),
                    )
                    acc = acc + pv
                    plgpu.barrier_arrive(v_consumed)

            acc_ref[batch_idx, split_idx, kv_head_idx, :, :] = acc.astype(jnp.float32)
            l_ref[batch_idx, split_idx, kv_head_idx, :] = l_i.astype(jnp.float32)
            m_ref[batch_idx, split_idx, kv_head_idx, :] = m_i.astype(jnp.float32)

    acc_grouped_padded, l_grouped_padded, m_grouped_padded = plgpu.kernel(
        kernel_entry,
        out_shape=[
            jax.ShapeDtypeStruct(
                (
                    plan.batch_size,
                    plan.k_splits,
                    plan.num_kv_heads,
                    padded_block_q,
                    plan.head_dim,
                ),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (
                    plan.batch_size,
                    plan.k_splits,
                    plan.num_kv_heads,
                    padded_block_q,
                ),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (
                    plan.batch_size,
                    plan.k_splits,
                    plan.num_kv_heads,
                    padded_block_q,
                ),
                jnp.float32,
            ),
        ],
        scratch_shapes=dict(
            q_smem=plgpu.SMEM(
                (padded_block_q, plan.head_dim),
                q.dtype,
                transforms=transforms,
            ),
            k_smem=plgpu.SMEM(
                (plan.block_kv, plan.head_dim),
                k_cache.dtype,
                transforms=transforms,
            ),
            v_smem=plgpu.SMEM(
                (plan.block_kv, plan.head_dim),
                v_cache.dtype,
                transforms=transforms,
            ),
            q_barrier=plgpu.Barrier(num_barriers=1),
            k_barrier=plgpu.Barrier(num_barriers=1),
            v_barrier=plgpu.Barrier(num_barriers=1),
            k_consumed=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=1,
            ),
            v_consumed=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=1,
            ),
        ),
        grid=(plan.batch_size, plan.num_kv_heads, plan.k_splits),
        grid_names=("batch", "kv_heads", "splits"),
        num_threads=2,
        thread_name="wg",
        compiler_params=compiler_params,
    )(
        q_grouped_padded,
        k_cache_flat,
        v_cache_flat,
        block_tables,
        context_lens,
        scale_arr,
    )

    acc_grouped = acc_grouped_padded[:, :, :, : plan.q_heads_per_kv_head, :]
    l_grouped = l_grouped_padded[:, :, :, : plan.q_heads_per_kv_head]
    m_grouped = m_grouped_padded[:, :, :, : plan.q_heads_per_kv_head]
    acc_partial = acc_grouped.reshape(
        plan.batch_size,
        plan.k_splits,
        plan.num_heads,
        plan.head_dim,
    )
    l_partial = l_grouped.reshape(plan.batch_size, plan.k_splits, plan.num_heads)
    m_partial = m_grouped.reshape(plan.batch_size, plan.k_splits, plan.num_heads)
    return acc_partial, l_partial, m_partial


@partial(jax.jit, static_argnames=("plan",))
def _compute_throughput_v2_partials(
    *,
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    plan: ThroughputV2Plan,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute bridge-free throughput_v2 partials without using the batched core."""
    if plan.partial_kernel == "row_partition_mosaic_v1":
        return _compute_throughput_v2_partials_mosaic(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            plan=plan,
        )
    return _compute_throughput_v2_partials_jax(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
        plan=plan,
    )


def paged_decode_attention_mosaic_throughput_v2(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    block_size: int,
    config: MosaicAttentionConfig | None = None,
    split_k: int = 0,
    prepared_metadata_cache: dict[tuple, object] | None = None,
) -> jax.Array:
    """Bridge-free throughput_v2 path: row-native partials plus device reduction."""
    _check_mosaic_available()

    batch_size, num_heads, head_dim = q.shape
    _, cache_block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if cache_block_size != block_size:
        raise ValueError(f"block_size mismatch: cache={cache_block_size} arg={block_size}")
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim} cache={cache_head_dim}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if block_tables.shape[1] == 0:
        return jnp.zeros_like(q)

    plan = build_paged_decode_throughput_v2_plan(
        q=q,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        config=config,
        split_k=split_k,
    )
    if prepared_metadata_cache is not None:
        plan = prepared_metadata_cache.setdefault(plan.metadata_cache_key, plan)

    acc_partial, l_partial, m_partial = _compute_throughput_v2_partials(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
        plan=plan,
    )
    out = reduce_partitioned_decode_partials(
        acc_partial,
        l_partial,
        m_partial,
        axis=1,
        family="throughput_v2",
        backend_override=plan.reduction_backend,
    )
    return out.astype(q.dtype)


# =============================================================================
# Prefill Attention Kernel (Variable-length sequences)
# =============================================================================

def prefill_attention_mosaic(
    q: jax.Array,           # [total_tokens, num_heads, head_dim]
    k: jax.Array,           # [total_tokens, num_kv_heads, head_dim]
    v: jax.Array,           # [total_tokens, num_kv_heads, head_dim]
    cu_seqlens: jax.Array,  # [batch_size + 1] cumulative sequence lengths
    max_seqlen: int,
    scale: float,
    config: MosaicAttentionConfig,
) -> jax.Array:
    """Prefill attention using true Mosaic GPU primitives.
    
    For prefill, each sequence has many tokens, naturally satisfying WGMMA M>=64.
    We use FlashAttention3's emit_pipeline_warp_specialized pattern.
    
    This is a simplified version - full implementation would use:
    - Causal masking
    - Variable-length sequence handling (GroupInfo pattern from ragged_dot)
    
    Args:
        q: Query tensor [total_tokens, num_heads, head_dim].
        k: Key tensor [total_tokens, num_kv_heads, head_dim].
        v: Value tensor [total_tokens, num_kv_heads, head_dim].
        cu_seqlens: Cumulative sequence lengths [batch_size + 1].
        max_seqlen: Maximum sequence length in batch.
        scale: Softmax scale (1/sqrt(head_dim)).
        config: Kernel configuration.
    
    Returns:
        Output tensor [total_tokens, num_heads, head_dim].
    """
    _check_mosaic_available()
    
    total_tokens, num_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    batch_size = cu_seqlens.shape[0] - 1
    
    block_q = config.block_q
    block_kv = config.block_kv
    num_compute_wgs = config.num_compute_wgs
    requested_steps = config.max_concurrent_steps
    
    # GQA ratio
    q_heads_per_kv_head = num_heads // num_kv_heads
    
    # Grid: (num_heads, num_q_tiles, batch_size)
    num_q_tiles = (max_seqlen + block_q * num_compute_wgs - 1) // (block_q * num_compute_wgs)
    
    # Compute SMEM transforms
    transforms = get_smem_transforms(head_dim, q.dtype)

    lse_bytes = num_compute_wgs * block_q * jnp.dtype(jnp.float32).itemsize
    meta_bytes = 0  # Prefill metadata remains small; adjust if padding is added later.
    max_concurrent_steps = _cap_pipeline_depth(
        block_q=block_q,
        block_kv=block_kv,
        head_dim=head_dim,
        dtype=q.dtype,
        num_compute_wgs=num_compute_wgs,
        requested_steps=requested_steps,
        extra_smem_bytes=lse_bytes,
        metadata_bytes=meta_bytes,
    )
    
    # Pre-computed row indices passed as kernel input to avoid 1D broadcasted_iota
    # which fails WG_STRIDED constraints when block_q < 128.
    # Pad to 128-element alignment for Mosaic layout rules.
    _row_indices_padded = _pad_last_dim_to_multiple(
        jnp.arange(block_q, dtype=jnp.int32)[None, :], multiple=128
    ).squeeze(0)

    def kernel_entry(q_ref, k_ref, v_ref, cu_seqlens_ref, row_indices_ref, out_ref, lse_ref):
        """Kernel entry with SMEM allocation."""

        qo_smem = plgpu.SMEM(
            (num_compute_wgs, block_q, head_dim),
            q.dtype,
            transforms=transforms,
        )
        k_smem = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            q.dtype,
            transforms=transforms,
        )
        v_smem = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            q.dtype,
            transforms=transforms,
        )
        lse_smem = plgpu.SMEM(
            (num_compute_wgs, block_q),
            jnp.float32,
        )
        # Alias LSE scratch with KV scratch to lower peak shared-memory usage:
        # they are used in disjoint phases (KV pipeline vs epilogue store).
        kv_lse_smem_union = plgpu.RefUnion(
            (k_smem, v_smem),
            (lse_smem,),
        )

        # Barriers
        k_barriers = plgpu.Barrier(num_barriers=max_concurrent_steps)
        v_barriers = plgpu.Barrier(num_barriers=max_concurrent_steps)
        q_barriers = plgpu.Barrier(num_barriers=num_compute_wgs)
        k_consumed = plgpu.Barrier(
            num_arrivals=num_compute_wgs,
            num_barriers=max_concurrent_steps,
        )
        v_consumed = plgpu.Barrier(
            num_arrivals=num_compute_wgs,
            num_barriers=max_concurrent_steps,
        )
        schedule_barrier = plgpu.Barrier(num_arrivals=num_compute_wgs)

        pl.run_scoped(
            lambda *args: prefill_kernel_body(
                q_ref, k_ref, v_ref, cu_seqlens_ref, row_indices_ref,
                out_ref, lse_ref, args
            ),
            (qo_smem, kv_lse_smem_union),
            (k_barriers, v_barriers, q_barriers),
            (k_consumed, v_consumed),
            schedule_barrier,
            collective_axes="wg",
        )
    
    def prefill_kernel_body(q_ref, k_ref, v_ref, cu_seqlens_ref, row_indices_ref, out_ref, lse_ref, scoped):
        """Prefill kernel body with warp specialization."""
        smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
        qo_smem, kv_lse_smem_union = smem_buffers
        ((k_smem, v_smem), (lse_smem,)) = kv_lse_smem_union
        k_barriers, v_barriers, q_barriers = buffer_barriers
        k_consumed, v_consumed = consumed_barriers
        
        # Grid indices
        head_idx = lax.axis_index("heads")
        q_tile_idx = lax.axis_index("q_seq")
        batch_idx = lax.axis_index("batch")
        wg_idx = lax.axis_index("wg")
        
        kv_head_idx = lax.div(head_idx, jnp.array(q_heads_per_kv_head, head_idx.dtype))
        
        # Get sequence boundaries
        seq_start = cu_seqlens_ref[batch_idx]
        seq_end = cu_seqlens_ref[batch_idx + 1]
        seq_len = (seq_end - seq_start).astype(jnp.int32)
        
        # KV loop iterations
        kv_seq_len = seq_len  # For prefill, K/V have same length as Q
        num_kv_steps = lax.div(kv_seq_len + block_kv - 1, jnp.array(block_kv, kv_seq_len.dtype))
        
        def perform_schedule_barrier():
            if config.use_schedule_barrier:
                plgpu.barrier_arrive(schedule_barrier)
                plgpu.barrier_wait(schedule_barrier)
        
        # ---------------------------------------------------------------------
        # Compute Warpgroups
        # ---------------------------------------------------------------------
        @pl.when(wg_idx < num_compute_wgs)
        def _compute_wg():
            plgpu.set_max_registers(232, action="increase")
            
            tile_group = jnp.int32(num_compute_wgs)
            global_tile_idx = q_tile_idx * tile_group + wg_idx
            tile_info = PrefillTileInfo.create(seq_len, block_q, global_tile_idx)
            tile_rows = tile_info.actual_size
            has_rows = tile_rows > 0

            @pl.when(has_rows)
            def _valid_q_tile():
                qo_tile_ref = qo_smem.at[wg_idx]
                qo_tile_ref[...] = jnp.zeros_like(qo_tile_ref[...])

                q_tile_start = seq_start + tile_info.actual_start
                smem_cursor = tile_info.start_within_block
                gmem_cursor = q_tile_start
                while_remaining = block_q
                while while_remaining > 0:
                    rows = 1 << int(math.log2(while_remaining))
                    while_remaining //= 2

                    @pl.when(tile_rows & rows != 0)
                    def _copy_tile(
                        smem_offset=smem_cursor,
                        gmem_offset=gmem_cursor,
                        rows=rows,
                    ):
                        plgpu.copy_gmem_to_smem(
                            q_ref.at[pl.ds(gmem_offset, rows), head_idx, :],
                            qo_tile_ref.at[pl.ds(smem_offset, rows)],
                            q_barriers.at[wg_idx],
                        )
                        plgpu.barrier_wait(q_barriers.at[wg_idx])

                    smem_cursor = smem_cursor + (tile_rows & rows)
                    gmem_cursor = gmem_cursor + (tile_rows & rows)

                # Row indices from pre-computed kernel input (avoids 1D broadcasted_iota
                # which fails WG_STRIDED constraints when block_q < 128).
                row_ids = row_indices_ref[:block_q]
                valid_start = tile_info.start_within_block
                valid_end = valid_start + tile_rows
                row_mask = (row_ids >= valid_start) & (row_ids < valid_end)
                row_mask_row = row_mask
                row_mask_row_f32 = row_mask.astype(jnp.float32)
                qo_tile_ref[...] = qo_tile_ref[...] * row_mask[:, None].astype(q.dtype)

                m_i = plgpu.layout_cast(
                    jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
                    _WGMMA_ROW,
                )
                l_i = plgpu.layout_cast(
                    jnp.full((block_q,), 0.0, dtype=jnp.float32),
                    _WGMMA_ROW,
                )
                acc = plgpu.layout_cast(
                    jnp.full((block_q, head_dim), 0.0, dtype=jnp.float32),
                    plgpu.Layout.WGMMA,
                )

                @pl.when(num_kv_steps > 0)
                def _wait_first_k():
                    plgpu.barrier_wait(k_barriers.at[0])

                pl.when(wg_idx == 1)(perform_schedule_barrier)

                def kv_loop(kv_step, carry):
                    acc, m_i, l_i = carry
                    slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))

                    def compute_qk(acc_ref):
                        plgpu.wgmma(
                            acc_ref,
                            qo_tile_ref,
                            plgpu.transpose_ref(k_smem.at[slot], (1, 0)),
                        )
                        plgpu.wgmma_wait(0)
                        perform_schedule_barrier()
                        return acc_ref[...]

                    qk = pl.run_scoped(
                        compute_qk,
                        plgpu.ACC((block_q, block_kv), jnp.float32),
                    )
                    plgpu.barrier_arrive(k_consumed.at[slot])

                    qk = qk * scale

                    q_ids = plgpu.layout_cast(
                        plgpu.broadcasted_iota(
                            jnp.int32, (block_q, block_kv), 0, layout=plgpu.Layout.WGMMA
                        ),
                        plgpu.Layout.WGMMA,
                    )
                    kv_ids = plgpu.layout_cast(
                        plgpu.broadcasted_iota(
                            jnp.int32, (block_q, block_kv), 1, layout=plgpu.Layout.WGMMA
                        ),
                        plgpu.Layout.WGMMA,
                    )
                    q_positions = tile_info.block_start + q_ids
                    kv_positions = kv_step * block_kv + kv_ids
                    causal_mask = q_positions >= kv_positions
                    valid_mask = kv_positions < kv_seq_len
                    mask = causal_mask & valid_mask & row_mask_row[:, None]
                    qk = jnp.where(mask, qk, -jnp.inf)

                    log2e = math.log2(math.e)
                    qk_max = qk.max(axis=1) * log2e
                    m_candidate = jnp.maximum(m_i, qk_max)
                    m_ij = jnp.where(row_mask_row, m_candidate, m_i)
                    # NaN-safe: -inf - (-inf) = NaN → replace with 0.0
                    safe_diff = jnp.where(m_i == m_ij, 0.0, m_i - m_ij)
                    alpha = jnp.where(row_mask_row, jnp.exp2(safe_diff), 1.0)
                    m_i = jnp.where(row_mask_row, m_ij, m_i)
                    p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                    p = jnp.where(mask, p, 0.0)
                    acc = acc * lax.broadcast_in_dim(alpha, acc.shape, [0])
                    l_i = l_i * alpha
                    p16 = p.astype(q.dtype)

                    perform_schedule_barrier()
                    plgpu.barrier_wait(v_barriers.at[slot])
                    l_i = jnp.where(row_mask_row, l_i + p.sum(axis=1), l_i)

                    def compute_pv(acc_ref):
                        plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
                        plgpu.wgmma_wait(0)
                        wait_step = kv_step + 1
                        wait_slot = lax.rem(wait_step, jnp.array(max_concurrent_steps, kv_step.dtype))

                        @pl.when(wait_step < num_kv_steps)
                        def _wait_next():
                            plgpu.barrier_wait(k_barriers.at[wait_slot])

                    acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                    plgpu.barrier_arrive(v_consumed.at[slot])

                    return acc, m_i, l_i

                acc, m_i, l_i = lax.fori_loop(
                    0, num_kv_steps.astype(jnp.int32), kv_loop, (acc, m_i, l_i)
                )

                pl.when(wg_idx == 0)(perform_schedule_barrier)

                safe_l = jnp.maximum(l_i, 1e-9)
                safe_l = jnp.where(row_mask_row_f32 > 0, safe_l, 1.0)
                acc = acc / lax.broadcast_in_dim(safe_l, (block_q, head_dim), [0])
                acc = jnp.where(row_mask_row[:, None], acc, 0.0)

                qo_tile_ref[...] = acc.astype(q.dtype)
                plgpu.commit_smem()

                def store_rows(dst_builder):
                    smem_cursor = tile_info.start_within_block
                    dst_cursor = tile_info.actual_start
                    remaining_rows = block_q
                    rows_mask = tile_rows
                    while remaining_rows > 0:
                        rows = 1 << int(math.log2(remaining_rows))
                        remaining_rows //= 2

                        @pl.when(rows_mask & rows != 0)
                        def _store_chunk(
                            smem_offset=smem_cursor,
                            dst_offset=dst_cursor,
                            rows=rows,
                        ):
                            plgpu.copy_smem_to_gmem(
                                qo_tile_ref.at[pl.ds(smem_offset, rows)],
                                dst_builder(dst_offset, rows),
                                commit_group=False,
                            )

                        smem_cursor = smem_cursor + (rows_mask & rows)
                        dst_cursor = dst_cursor + (rows_mask & rows)

                @pl.when(tile_rows > 0)
                def _store_output():
                    def out_builder(offset, rows):
                        return out_ref.at[
                            pl.ds(seq_start + offset, rows),
                            head_idx,
                            :,
                        ]

                    store_rows(out_builder)
                    plgpu.commit_smem_to_gmem_group()

                RCP_LN2 = 1.4426950408889634
                log2_fn = lambda x: jnp.log(jnp.maximum(x, 1e-9)) * RCP_LN2
                lse = jnp.where(row_mask_row, m_i + log2_fn(l_i), -jnp.inf)
                lse_tile_ref = lse_smem.at[wg_idx]
                lse_tile_ref[...] = lse
                plgpu.commit_smem()

                @pl.when(tile_rows > 0)
                def _store_lse():
                    def lse_builder(offset, rows):
                        return lse_ref.at[
                            batch_idx,
                            head_idx,
                            pl.ds(offset, rows),
                        ]

                    smem_cursor = tile_info.start_within_block
                    dst_cursor = tile_info.actual_start
                    remaining_rows = block_q
                    rows_mask = tile_rows
                    while remaining_rows > 0:
                        rows = 1 << int(math.log2(remaining_rows))
                        remaining_rows //= 2

                        @pl.when(rows_mask & rows != 0)
                        def _store_chunk(
                            smem_offset=smem_cursor,
                            dst_offset=dst_cursor,
                            rows=rows,
                        ):
                            plgpu.copy_smem_to_gmem(
                                lse_tile_ref.at[pl.ds(smem_offset, rows)],
                                lse_builder(dst_offset, rows),
                                commit_group=False,
                            )

                        smem_cursor = smem_cursor + (rows_mask & rows)
                        dst_cursor = dst_cursor + (rows_mask & rows)
                    plgpu.commit_smem_to_gmem_group()

                plgpu.wait_smem_to_gmem(0)
        
        # ---------------------------------------------------------------------
        # Memory Warpgroup
        # ---------------------------------------------------------------------
        @pl.when(wg_idx == num_compute_wgs)
        def _memory_wg():
            plgpu.set_max_registers(40, action="decrease")
            
            kv_global_start = seq_start
            
            # Prefill pipeline
            for i in range(max_concurrent_steps):
                kv_pos = kv_global_start + i * block_kv
                plgpu.copy_gmem_to_smem(
                    k_ref.at[pl.ds(kv_pos, block_kv), kv_head_idx, :],
                    k_smem.at[i],
                    k_barriers.at[i],
                )
                plgpu.copy_gmem_to_smem(
                    v_ref.at[pl.ds(kv_pos, block_kv), kv_head_idx, :],
                    v_smem.at[i],
                    v_barriers.at[i],
                )
            
            @pl.loop(0, num_kv_steps - max_concurrent_steps)
            def _stream_loop(kv_step):
                tma_step = kv_step + max_concurrent_steps
                tma_slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                kv_pos = kv_global_start + tma_step * block_kv
                
                plgpu.barrier_wait(k_consumed.at[tma_slot])
                plgpu.copy_gmem_to_smem(
                    k_ref.at[pl.ds(kv_pos, block_kv), kv_head_idx, :],
                    k_smem.at[tma_slot],
                    k_barriers.at[tma_slot],
                )
                
                plgpu.barrier_wait(v_consumed.at[tma_slot])
                plgpu.copy_gmem_to_smem(
                    v_ref.at[pl.ds(kv_pos, block_kv), kv_head_idx, :],
                    v_smem.at[tma_slot],
                    v_barriers.at[tma_slot],
                )
    
    # Output shape includes LSE for backward pass
    out_shape = [
        jax.ShapeDtypeStruct((total_tokens, num_heads, head_dim), q.dtype),
        jax.ShapeDtypeStruct((batch_size, num_heads, max_seqlen), jnp.float32),
    ]
    
    return plgpu.kernel(
        kernel_entry,
        out_shape=out_shape,
        grid=(num_heads, num_q_tiles, batch_size),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=num_compute_wgs + 1,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(q, k, v, cu_seqlens, _row_indices_padded)


# =============================================================================
# High-Level API
# =============================================================================


def prefill_attention_mosaic_api(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    max_seqlen: int,
    scale: float,
    config: MosaicAttentionConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
    """High-level prefill attention API using Mosaic GPU kernels.
    
    Args:
        q: Query tensor [total_tokens, num_heads, head_dim].
        k: Key tensor [total_tokens, num_kv_heads, head_dim].
        v: Value tensor [total_tokens, num_kv_heads, head_dim].
        cu_seqlens: Cumulative sequence lengths [batch_size + 1].
        max_seqlen: Maximum sequence length.
        scale: Softmax scale.
        config: Optional kernel configuration.
    
    Returns:
        Tuple of (output, lse):
        - output: [total_tokens, num_heads, head_dim]
        - lse: [batch_size, num_heads, max_seqlen] log-sum-exp for backward
    """
    if config is None:
        config = MosaicAttentionConfig()
    
    return prefill_attention_mosaic(q, k, v, cu_seqlens, max_seqlen, scale, config)
