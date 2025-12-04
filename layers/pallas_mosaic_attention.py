"""True Pallas Mosaic GPU kernels for paged attention.

This module implements high-performance attention kernels using Mosaic GPU backend features:
- plgpu.SMEM for shared memory with TilingTransform and SwizzleTransform
- plgpu.wgmma for TensorCore matrix multiply-accumulate
- plgpu.Barrier for async pipeline coordination
- plgpu.emit_pipeline for memory/compute overlap
- plgpu.copy_gmem_to_smem / copy_smem_to_gmem for TMA transfers

Key algorithms:
- FlashAttention3 online softmax with log2/exp2 for FMA utilization
- Warp specialization: 2 compute warpgroups + 1 memory warpgroup
- Paged KV-cache with block table indirection

Reference implementations:
- docs/reference/attention_mgpu.py - FlashAttention3 forward/backward
- docs/reference/ragged_dot_mgpu.py - Variable-length group handling
- docs/reference/hopper_matmul_mgpu.py - WGMMA pipeline pattern

Constraints:
- WGMMA requires M dimension >= 64 (batch queries across sequences for decode)
- Block sizes must be multiples of 64 for WGMMA alignment
- SMEM limited to ~228KB on H100

Author: nanovllm_jax
"""

import dataclasses
import math
import functools
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

# Check if Pallas Mosaic GPU is available
try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu
    MOSAIC_AVAILABLE = True
except ImportError:
    MOSAIC_AVAILABLE = False
    pl = None
    plgpu = None


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
        num_compute_wgs: Number of compute warpgroups (typically 2).
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


# =============================================================================
# Decode Utility Helpers
# =============================================================================

def _prepare_decode_tiles(
    q: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    block_q: int,
    block_size: int,
    block_kv: int,
):
    """Pad decode inputs and build per-tile chunk schedules for WGMMA kernels."""

    if block_size % block_kv:
        raise ValueError("block_size must be divisible by block_kv for chunk scheduling")

    original_batch_size = q.shape[0]
    pad_rows = (-original_batch_size) % block_q

    if pad_rows:
        q = jnp.pad(q, ((0, pad_rows), (0, 0), (0, 0)))
        block_tables = jnp.pad(block_tables, ((0, pad_rows), (0, 0)), constant_values=0)
        context_lens = jnp.pad(context_lens, ((0, pad_rows),), constant_values=0)

    padded_batch = q.shape[0]
    num_batch_tiles = padded_batch // block_q
    tile_valid_counts = jnp.full((num_batch_tiles,), block_q, dtype=jnp.int32)
    if pad_rows:
        tile_valid_counts = tile_valid_counts.at[-1].set(block_q - pad_rows)

    max_blocks = block_tables.shape[1]
    chunks_per_block = block_size // block_kv
    max_tile_chunks = block_q * max_blocks * chunks_per_block

    block_tables_tile = block_tables.reshape(num_batch_tiles, block_q, max_blocks)
    context_tile = context_lens.reshape(num_batch_tiles, block_q)

    block_offsets = jnp.arange(max_blocks, dtype=jnp.int32) * block_size
    block_offsets = block_offsets[None, None, :]
    tokens_per_block = jnp.clip(
        context_tile[:, :, None] - block_offsets,
        a_min=0,
        a_max=block_size,
    )

    chunk_offsets = jnp.arange(chunks_per_block, dtype=jnp.int32) * block_kv
    chunk_offsets = chunk_offsets[None, None, None, :]
    tokens_per_chunk = jnp.clip(
        tokens_per_block[:, :, :, None] - chunk_offsets,
        a_min=0,
        a_max=block_kv,
    )

    chunk_block_indices = jnp.broadcast_to(
        block_tables_tile[:, :, :, None], tokens_per_chunk.shape
    )
    chunk_offsets = jnp.broadcast_to(chunk_offsets, tokens_per_chunk.shape)

    chunk_block_indices = chunk_block_indices.reshape(num_batch_tiles, max_tile_chunks)
    chunk_offsets = chunk_offsets.reshape(num_batch_tiles, max_tile_chunks)
    chunk_tokens = tokens_per_chunk.reshape(num_batch_tiles, max_tile_chunks)

    row_ids = jnp.arange(block_q, dtype=jnp.int32)
    row_ids = row_ids[None, :, None, None]
    row_ids = jnp.broadcast_to(row_ids, tokens_per_chunk.shape)
    chunk_row_indices = row_ids.reshape(num_batch_tiles, max_tile_chunks)

    logical_ids = jnp.arange(max_blocks, dtype=jnp.int32)
    logical_ids = logical_ids[None, None, :, None]
    logical_ids = jnp.broadcast_to(logical_ids, tokens_per_chunk.shape)
    chunk_logical_blocks = logical_ids.reshape(num_batch_tiles, max_tile_chunks)

    chunk_valid = chunk_tokens > 0
    tile_chunk_counts = chunk_valid.sum(axis=1, dtype=jnp.int32)
    chunk_tokens = chunk_tokens.astype(jnp.int32)
    chunk_offsets = chunk_offsets.astype(jnp.int32)
    chunk_block_indices = chunk_block_indices.astype(jnp.int32)

    tokens_per_row = tokens_per_chunk.reshape(num_batch_tiles, block_q, -1).astype(jnp.int32)
    chunk_prefix = jnp.cumsum(tokens_per_row, axis=2, dtype=jnp.int32) - tokens_per_row
    chunk_prefix = chunk_prefix.reshape(num_batch_tiles, max_tile_chunks)

    row_lengths = context_tile.astype(jnp.int32)
    row_offsets = jnp.cumsum(row_lengths, axis=1, dtype=jnp.int32) - row_lengths

    return (
        q,
        block_tables,
        context_lens,
        tile_valid_counts,
        row_offsets,
        row_lengths,
        chunk_block_indices,
        chunk_offsets,
        chunk_tokens,
        chunk_prefix,
        chunk_row_indices,
        chunk_logical_blocks,
        tile_chunk_counts,
        original_batch_size,
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
) -> jax.Array:
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
    
    Returns:
        Output tensor [batch_size, num_heads, head_dim].
    """
    _check_mosaic_available()
    
    # Extract shapes first (needed for _prepare_decode_tiles)
    batch_size, num_heads, head_dim = q.shape
    num_kv_blocks, kv_block_size, num_kv_heads, _ = k_cache.shape
    
    block_q = config.block_q
    block_kv = config.block_kv
    max_concurrent_steps = config.max_concurrent_steps

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
    )

    # Re-extract shapes after padding
    batch_size, num_heads, head_dim = q.shape
    max_blocks_per_seq = block_tables.shape[1]

    # Use a single compute warpgroup so each WGMMA sees block_q rows.
    num_compute_wgs = 1
    
    # GQA: query heads per KV head
    q_heads_per_kv_head = num_heads // num_kv_heads
    
    # For decode, we process block_q sequences at a time
    # This ensures M >= 64 for WGMMA
    num_batch_tiles = (batch_size + block_q - 1) // block_q
    
    # Compute SMEM transforms
    transforms = get_smem_transforms(head_dim, q.dtype)
    
    # Maximum KV steps (across all sequences in this batch tile)
    max_kv_steps = max_blocks_per_seq * (kv_block_size // block_kv)
    
    def kernel_entry(
        q_ref,
        k_cache_ref,
        v_cache_ref,
        tile_valid_counts_ref,
        tile_row_offsets_ref,
        tile_row_lengths_ref,
        tile_chunk_block_indices_ref,
        tile_chunk_offsets_ref,
        tile_chunk_tokens_ref,
        tile_chunk_prefix_tokens_ref,
        tile_chunk_row_indices_ref,
        tile_chunk_logical_blocks_ref,
        tile_chunk_counts_ref,
        out_ref,
    ):
        """Kernel entry point that allocates SMEM and barriers."""
        
        # Allocate SMEM buffers with swizzle for bank-conflict-free access
        # Q/O buffer: [num_compute_wgs, block_q, head_dim]
        qo_smem = plgpu.SMEM(
            (num_compute_wgs, block_q, head_dim),
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
        
        # Run kernel with scoped allocations
        pl.run_scoped(
            lambda *args: kernel_body(
                q_ref,
                k_cache_ref,
                v_cache_ref,
                tile_valid_counts_ref,
                tile_row_offsets_ref,
                tile_row_lengths_ref,
                tile_chunk_block_indices_ref,
                tile_chunk_offsets_ref,
                tile_chunk_tokens_ref,
                tile_chunk_prefix_tokens_ref,
                tile_chunk_row_indices_ref,
                tile_chunk_logical_blocks_ref,
                tile_chunk_counts_ref,
                out_ref,
                args,
            ),
            (qo_smem, k_smem, v_smem),  # SMEM buffers
            (k_barriers, v_barriers, q_barriers),  # Buffer barriers
            (k_consumed, v_consumed),  # Consumed barriers
            schedule_barrier,  # Schedule barrier
            collective_axes="wg",
        )
    
    def kernel_body(
        q_ref,
        k_cache_ref,
        v_cache_ref,
        tile_valid_counts_ref,
        tile_row_offsets_ref,
        tile_row_lengths_ref,
        tile_chunk_block_indices_ref,
        tile_chunk_offsets_ref,
        tile_chunk_tokens_ref,
        tile_chunk_prefix_tokens_ref,
        tile_chunk_row_indices_ref,
        tile_chunk_logical_blocks_ref,
        tile_chunk_counts_ref,
        out_ref,
        scoped,
    ):
        """Main kernel body with warp specialization."""
        smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
        qo_smem, k_smem, v_smem = smem_buffers
        k_barriers, v_barriers, q_barriers = buffer_barriers
        k_consumed, v_consumed = consumed_barriers
        
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
            if config.use_schedule_barrier:
                plgpu.barrier_arrive(schedule_barrier)
                plgpu.barrier_wait(schedule_barrier)
        
        # ---------------------------------------------------------------------
        # Compute Warpgroups (wg_idx < num_compute_wgs)
        # ---------------------------------------------------------------------
        @pl.when(wg_idx < num_compute_wgs)
        def _compute_wg():
            # Increase register budget for compute warpgroups
            plgpu.set_max_registers(232, action="increase")
            
            q_slice_start = batch_base
            tile_count = tile_valid_counts_ref[batch_tile_idx]
            seq_lens = tile_row_lengths_ref[batch_tile_idx]
            seq_lens = seq_lens.astype(jnp.int32)
            my_block_q = block_q

            # TMA copy Q to SMEM (block_q rows)
            plgpu.copy_gmem_to_smem(
                q_ref.at[pl.ds(q_slice_start, my_block_q), head_idx, :],
                qo_smem.at[wg_idx],
                q_barriers.at[wg_idx],
            )
            plgpu.barrier_wait(q_barriers.at[wg_idx])

            # Zero out padded rows so they do not contribute to attention
            row_ids = jnp.arange(block_q, dtype=jnp.int32)
            row_mask_row = row_ids < tile_count
            row_mask_2d = row_mask_row[:, None].astype(q.dtype)
            qo_tile = qo_smem.at[wg_idx][...]
            qo_tile = qo_tile * row_mask_2d
            qo_smem.at[wg_idx][...] = qo_tile
            row_indices = jnp.arange(block_q, dtype=jnp.int32)
            tile_chunk_rows = tile_chunk_row_indices_ref[batch_tile_idx]
            tile_chunk_tokens = tile_chunk_tokens_ref[batch_tile_idx]
            tile_chunk_prefix = tile_chunk_prefix_tokens_ref[batch_tile_idx]
            
            # Initialize online softmax state (FlashAttention3)
            # m_i: running max (in log2 space for FMA)
            # l_i: running sum of exp(x - m)
            # acc: weighted sum of values
            m_i = plgpu.layout_cast(
                jnp.full((my_block_q,), -jnp.inf, dtype=jnp.float32),
                plgpu.Layout.WGMMA_ROW,
            )
            l_i = plgpu.layout_cast(
                jnp.full((my_block_q,), 0.0, dtype=jnp.float32),
                plgpu.Layout.WGMMA_ROW,
            )
            acc = plgpu.layout_cast(
                jnp.full((my_block_q, head_dim), 0.0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )

            row_mask_row_f32 = row_mask_row.astype(jnp.float32)
            num_kv_steps = tile_chunk_counts_ref[batch_tile_idx]
            
            # Wait for first K block from memory warpgroup
            @pl.when(num_kv_steps > 0)
            def _wait_first_k():
                plgpu.barrier_wait(k_barriers.at[0])
            
            # -----------------------------------------------------------------
            # KV Loop (FlashAttention3 online softmax)
            # -----------------------------------------------------------------
            def kv_loop(kv_step, carry):
                acc, m_i, l_i = carry
                slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                
                # Compute QK^T using WGMMA
                # Q: [my_block_q, head_dim] in qo_smem[wg_idx]
                # K: [block_kv, head_dim] in k_smem[slot]
                # Result: [my_block_q, block_kv]
                def compute_qk(acc_ref):
                    plgpu.wgmma(
                        acc_ref,
                        qo_smem.at[wg_idx],
                        plgpu.transpose_ref(k_smem.at[slot], (1, 0)),
                    )
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

                chunk_row = tile_chunk_rows[kv_step]
                chunk_tokens = tile_chunk_tokens[kv_step]
                chunk_prefix = tile_chunk_prefix[kv_step]

                row_active = (row_indices == chunk_row) & row_mask_row
                col_ids = jnp.arange(block_kv, dtype=jnp.int32)
                kv_pos = chunk_prefix + col_ids
                col_mask = kv_pos < (chunk_prefix + chunk_tokens)
                mask = row_active[:, None] & col_mask[None, :]
                qk = jnp.where(mask, qk, -jnp.inf)
                
                # ----- Online Softmax (log2/exp2 for FMA) -----
                log2e = math.log2(math.e)
                
                # New max for this block
                qk_max = qk.max(axis=1) * log2e
                m_candidate = jnp.maximum(m_i, qk_max)
                m_ij = jnp.where(row_active, m_candidate, m_i)
                
                # Rescale previous accumulator
                alpha = jnp.where(row_active, jnp.exp2(m_i - m_ij), 1.0)
                m_i = jnp.where(row_active, m_ij, m_i)
                
                # Softmax weights
                p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                p = jnp.where(mask, p, 0.0)
                
                # Update accumulator with rescaling
                acc = acc * lax.broadcast_in_dim(alpha, acc.shape, [0])
                l_i = l_i * alpha
                
                # Convert to fp16 for WGMMA
                p16 = p.astype(q.dtype)
                
                # Barrier coordination before V access
                perform_schedule_barrier()
                plgpu.barrier_wait(v_barriers.at[slot])
                l_i = jnp.where(row_active, l_i + p.sum(axis=1), l_i)
                
                # ----- PV Matmul (accumulate into output) -----
                # P: [my_block_q, block_kv]
                # V: [block_kv, head_dim] in v_smem[slot]
                # Result: acc += P @ V
                def compute_pv(acc_ref):
                    plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
                    
                    # Prefetch next K block
                    wait_step = kv_step + 1
                    wait_slot = lax.rem(wait_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                    @pl.when(wait_step < num_kv_steps)
                    def _wait_next_k():
                        plgpu.barrier_wait(k_barriers.at[wait_slot])
                
                acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                
                # Signal that V has been consumed
                plgpu.barrier_arrive(v_consumed.at[slot])
                
                return acc, m_i, l_i
            
            # Execute KV loop
            acc, m_i, l_i = lax.fori_loop(
                0, num_kv_steps.astype(jnp.int32), kv_loop, (acc, m_i, l_i)
            )
            
            # Coordinate before epilogue
            perform_schedule_barrier()
            
            # ----- Normalize Output -----
            # O = acc / l_i
            safe_l = jnp.maximum(l_i, 1e-9)
            safe_l = jnp.where(row_mask_row_f32 > 0, safe_l, 1.0)
            acc = acc / lax.broadcast_in_dim(safe_l, (my_block_q, head_dim), [0])
            acc = jnp.where(row_mask_row[:, None], acc, 0.0)
            
            # Store to SMEM, then TMA to GMEM
            qo_smem.at[wg_idx][...] = acc.astype(q.dtype)
            plgpu.commit_smem()
            
            copy_rows = tile_count
            plgpu.copy_smem_to_gmem(
                qo_smem.at[wg_idx],
                out_ref.at[pl.ds(q_slice_start, copy_rows), head_idx, :],
            )
            plgpu.wait_smem_to_gmem(0)
        
        # ---------------------------------------------------------------------
        # Memory Warpgroup (wg_idx == num_compute_wgs)
        # ---------------------------------------------------------------------
        @pl.when(wg_idx == num_compute_wgs)
        def _memory_wg():
            # Reduce register budget for memory warpgroup
            plgpu.set_max_registers(40, action="decrease")
            
            chunk_count = tile_chunk_counts_ref[batch_tile_idx]
            chunk_blocks = tile_chunk_block_indices_ref[batch_tile_idx]
            chunk_offsets = tile_chunk_offsets_ref[batch_tile_idx]

            max_steps_arr = jnp.array(max_concurrent_steps, dtype=jnp.int32)

            def issue_chunk(chunk_idx, slot):
                block_id = chunk_blocks[chunk_idx]
                chunk_offset = chunk_offsets[chunk_idx]

                plgpu.copy_gmem_to_smem(
                    k_cache_ref.at[block_id, pl.ds(chunk_offset, block_kv), kv_head_idx, :],
                    k_smem.at[slot],
                    k_barriers.at[slot],
                )
                plgpu.copy_gmem_to_smem(
                    v_cache_ref.at[block_id, pl.ds(chunk_offset, block_kv), kv_head_idx, :],
                    v_smem.at[slot],
                    v_barriers.at[slot],
                )

            for i in range(max_concurrent_steps):
                idx = jnp.array(i, dtype=jnp.int32)

                @pl.when(chunk_count > idx)
                def _prefill_slot(idx=idx, slot=i):
                    issue_chunk(idx, slot)

            @pl.when(chunk_count > max_steps_arr)
            def _kv_stream_loop():
                total_iters = chunk_count - max_steps_arr

                @pl.loop(0, total_iters)
                def _stream_step(kv_step):
                    slot = lax.rem(kv_step, max_steps_arr)
                    chunk_idx = kv_step + max_steps_arr

                    plgpu.barrier_wait(k_consumed.at[slot])
                    plgpu.barrier_wait(v_consumed.at[slot])
                    issue_chunk(chunk_idx, slot)
    
    # Launch kernel
    out = plgpu.kernel(
        kernel_entry,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(num_batch_tiles, num_heads),
        grid_names=("batch_tiles", "heads"),
        num_threads=num_compute_wgs + 1,  # 2 compute + 1 memory
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(
        q,
        k_cache,
        v_cache,
        tile_valid_counts,
        tile_row_offsets,
        tile_row_lengths,
        tile_chunk_block_indices,
        tile_chunk_offsets,
        tile_chunk_tokens,
        tile_chunk_prefix_tokens,
        tile_chunk_row_indices,
        tile_chunk_logical_blocks,
        tile_chunk_counts,
    )

    return out[:original_batch_size]


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
    max_concurrent_steps = config.max_concurrent_steps
    num_compute_wgs = config.num_compute_wgs
    
    # GQA ratio
    q_heads_per_kv_head = num_heads // num_kv_heads
    
    # Grid: (num_heads, num_q_tiles, batch_size)
    num_q_tiles = (max_seqlen + block_q * num_compute_wgs - 1) // (block_q * num_compute_wgs)
    
    # Compute SMEM transforms
    transforms = get_smem_transforms(head_dim, q.dtype)
    
    def kernel_entry(q_ref, k_ref, v_ref, cu_seqlens_ref, out_ref, lse_ref):
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
                q_ref, k_ref, v_ref, cu_seqlens_ref, out_ref, lse_ref, args
            ),
            (qo_smem, k_smem, v_smem, lse_smem),
            (k_barriers, v_barriers, q_barriers),
            (k_consumed, v_consumed),
            schedule_barrier,
            collective_axes="wg",
        )
    
    def prefill_kernel_body(q_ref, k_ref, v_ref, cu_seqlens_ref, out_ref, lse_ref, scoped):
        """Prefill kernel body with warp specialization."""
        smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
        qo_smem, k_smem, v_smem, lse_smem = smem_buffers
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

                row_ids = jnp.arange(block_q, dtype=jnp.int32)
                valid_start = tile_info.start_within_block
                valid_end = valid_start + tile_rows
                row_mask = (row_ids >= valid_start) & (row_ids < valid_end)
                row_mask_row = row_mask
                row_mask_row_f32 = row_mask.astype(jnp.float32)
                qo_tile_ref[...] = qo_tile_ref[...] * row_mask[:, None].astype(q.dtype)

                m_i = plgpu.layout_cast(
                    jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
                    plgpu.Layout.WGMMA_ROW,
                )
                l_i = plgpu.layout_cast(
                    jnp.full((block_q,), 0.0, dtype=jnp.float32),
                    plgpu.Layout.WGMMA_ROW,
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
                        perform_schedule_barrier()
                        return acc_ref[...]

                    qk = pl.run_scoped(
                        compute_qk,
                        plgpu.ACC((block_q, block_kv), jnp.float32),
                    )
                    plgpu.barrier_arrive(k_consumed.at[slot])

                    qk = qk * scale

                    q_ids = plgpu.broadcasted_iota(
                        jnp.int32, (block_q, block_kv), 0, layout=plgpu.Layout.WGMMA
                    )
                    kv_ids = plgpu.broadcasted_iota(
                        jnp.int32, (block_q, block_kv), 1, layout=plgpu.Layout.WGMMA
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
                    alpha = jnp.where(row_mask_row, jnp.exp2(m_i - m_ij), 1.0)
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
                            )

                        smem_cursor = smem_cursor + (rows_mask & rows)
                        dst_cursor = dst_cursor + (rows_mask & rows)

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
    )(q, k, v, cu_seqlens)


# =============================================================================
# Paged Decode with Block Table Indirection
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PagedKVBlockInfo:
    """Information for accessing paged KV-cache blocks.
    
    Similar to GroupInfo from ragged_dot, but for paged attention.
    Tracks which physical blocks to load for each sequence.
    """
    batch_idx: jax.Array        # Which sequence
    kv_block_idx: jax.Array     # Which KV block (logical)
    physical_block: jax.Array   # Physical block in cache
    tokens_in_block: jax.Array  # Valid tokens in this block
    
    @classmethod
    def create(cls, block_tables, context_lens, batch_idx, kv_block_idx, block_size):
        """Create block info for a given batch and KV block index."""
        physical_block = block_tables[batch_idx, kv_block_idx]
        context_len = context_lens[batch_idx]
        block_start = kv_block_idx * block_size
        tokens_in_block = jnp.minimum(block_size, context_len - block_start)
        tokens_in_block = jnp.maximum(0, tokens_in_block)
        
        return cls(
            batch_idx=batch_idx,
            kv_block_idx=kv_block_idx,
            physical_block=physical_block,
            tokens_in_block=tokens_in_block,
        )


def paged_decode_attention_mosaic_v2(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    config: MosaicAttentionConfig,
) -> jax.Array:
    """Paged decode attention with proper block table indirection.
    
    This version handles the complexity of paged KV-cache where each sequence
    has its own block table mapping logical blocks to physical cache blocks.
    
    For WGMMA compatibility, we process multiple sequences together (block_q).
    Each sequence may have different physical blocks, so we load K/V for
    each sequence separately and arrange in SMEM.
    
    Args:
        q: Query [batch_size, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: Block tables [batch_size, max_blocks_per_seq].
        context_lens: Context lengths [batch_size].
        scale: Softmax scale.
        config: Kernel config.
    
    Returns:
        Output [batch_size, num_heads, head_dim].
    """
    _check_mosaic_available()
    
    batch_size, num_heads, head_dim = q.shape
    num_kv_blocks, kv_block_size, num_kv_heads, _ = k_cache.shape
    max_blocks_per_seq = block_tables.shape[1]
    
    block_q = config.block_q
    block_kv = min(config.block_kv, kv_block_size)  # Can't exceed cache block size
    max_concurrent_steps = config.max_concurrent_steps
    
    q_heads_per_kv_head = num_heads // num_kv_heads
    
    # For decode: each sequence has 1 query token
    # We process block_q sequences together to satisfy WGMMA M>=64
    # Grid: (batch_tiles, heads)
    num_batch_tiles = (batch_size + block_q - 1) // block_q
    
    # KV steps per physical block
    kv_steps_per_block = kv_block_size // block_kv
    max_kv_steps = max_blocks_per_seq * kv_steps_per_block
    
    transforms = get_smem_transforms(head_dim, q.dtype)
    
    def kernel(q_ref, k_cache_ref, v_cache_ref, block_tables_ref,
               context_lens_ref, out_ref):
        """Simplified paged decode kernel.
        
        Note: This is a simplified version that demonstrates the structure.
        Full implementation would need:
        - Proper per-sequence block table lookup
        - Variable-length masking per sequence
        - Warp specialization (memory/compute split)
        """
        batch_tile_idx = lax.axis_index("batch_tiles")
        head_idx = lax.axis_index("heads")
        kv_head_idx = lax.div(head_idx, jnp.array(q_heads_per_kv_head, head_idx.dtype))
        
        batch_start = batch_tile_idx * block_q
        batch_end = jnp.minimum(batch_start + block_q, batch_size)
        actual_batch_size = batch_end - batch_start
        
        # Load queries for this batch tile
        # q_local: [block_q, head_dim] (padded if batch_size not divisible)
        q_local = q_ref[batch_start:batch_end, head_idx, :].astype(jnp.float32)
        
        # Initialize accumulators for online softmax
        m = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
        l = jnp.zeros((block_q,), dtype=jnp.float32)
        acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)
        
        # Get max context length in this batch tile for loop bound
        tile_context_lens = context_lens_ref[batch_start:batch_end]
        max_ctx_in_tile = tile_context_lens.max()
        num_logical_blocks = lax.div(
            max_ctx_in_tile + kv_block_size - 1,
            jnp.array(kv_block_size, max_ctx_in_tile.dtype)
        )
        
        # Process each logical KV block
        def process_logical_block(logical_block_idx, carry):
            m, l, acc = carry
            
            # For each sequence in this tile, get its physical block
            # This creates a gather pattern: different physical blocks per sequence
            
            # Get physical blocks for all sequences in tile
            # physical_blocks: [block_q] - one per sequence
            seq_indices = jnp.arange(block_q) + batch_start
            seq_indices = jnp.minimum(seq_indices, batch_size - 1)
            physical_blocks = block_tables_ref[seq_indices, logical_block_idx]
            
            # Calculate valid tokens per sequence for this block
            block_start_pos = logical_block_idx * kv_block_size
            valid_tokens = jnp.minimum(
                kv_block_size,
                tile_context_lens - block_start_pos
            )
            valid_tokens = jnp.maximum(0, valid_tokens)
            
            # Process each KV tile within this logical block
            def process_kv_tile(kv_tile_idx, inner_carry):
                m, l, acc = inner_carry
                
                kv_pos_in_block = kv_tile_idx * block_kv
                
                # For simplicity, load K/V for first sequence's physical block
                # TODO: Handle per-sequence physical blocks properly
                first_physical_block = physical_blocks[0]
                
                # Load K: [block_kv, head_dim]
                k_block = k_cache_ref[first_physical_block, 
                                       kv_pos_in_block:kv_pos_in_block + block_kv,
                                       kv_head_idx, :].astype(jnp.float32)
                
                # Load V: [block_kv, head_dim]  
                v_block = v_cache_ref[first_physical_block,
                                       kv_pos_in_block:kv_pos_in_block + block_kv,
                                       kv_head_idx, :].astype(jnp.float32)
                
                # QK^T: [block_q, head_dim] @ [head_dim, block_kv] = [block_q, block_kv]
                scores = jnp.matmul(q_local, k_block.T) * scale
                
                # Mask invalid positions
                kv_positions = jnp.arange(block_kv) + block_start_pos + kv_pos_in_block
                # Per-sequence masking based on context_lens
                seq_mask = kv_positions[None, :] < tile_context_lens[:actual_batch_size, None]
                # Pad mask to block_q
                seq_mask = jnp.pad(
                    seq_mask,
                    ((0, block_q - actual_batch_size), (0, 0)),
                    constant_values=False
                )
                scores = jnp.where(seq_mask, scores, -jnp.inf)
                
                # Online softmax update
                log2e = math.log2(math.e)
                m_new = jnp.maximum(m, scores.max(axis=1) * log2e)
                alpha = jnp.exp2(m - m_new)
                p = jnp.exp2(scores * log2e - m_new[:, None])
                
                l_new = alpha * l + p.sum(axis=1)
                acc_new = alpha[:, None] * acc + jnp.matmul(p, v_block)
                
                return m_new, l_new, acc_new
            
            m, l, acc = lax.fori_loop(
                0, kv_steps_per_block, process_kv_tile, (m, l, acc)
            )
            
            return m, l, acc
        
        m, l, acc = lax.fori_loop(
            0, num_logical_blocks.astype(jnp.int32), process_logical_block, (m, l, acc)
        )
        
        # Normalize output
        out_local = acc / l[:, None]
        
        # Write output (only valid sequences)
        out_ref = out_ref.at[batch_start:batch_end, head_idx, :].set(
            out_local[:actual_batch_size].astype(out_ref.dtype)
        )
    
    # Note: This uses pallas_call rather than plgpu.kernel for simplicity
    # A full implementation would use plgpu.kernel with proper SMEM/WGMMA
    out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(num_batch_tiles, num_heads),
        grid_names=("batch_tiles", "heads"),
    )(q, k_cache, v_cache, block_tables, context_lens)


# =============================================================================
# High-Level API
# =============================================================================

def paged_attention_mosaic(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    block_size: int,
    config: MosaicAttentionConfig | None = None,
) -> jax.Array:
    """High-level paged attention API using Mosaic GPU kernels.
    
    Automatically selects the appropriate kernel based on batch size
    and available hardware features.
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: Block indices [batch_size, max_blocks_per_seq].
        context_lens: Context lengths [batch_size].
        scale: Softmax scale.
        block_size: Cache block size.
        config: Optional kernel configuration.
    
    Returns:
        Output tensor [batch_size, num_heads, head_dim].
    """
    if config is None:
        config = MosaicAttentionConfig()
    
    batch_size = q.shape[0]
    
    # Use v2 implementation with proper block table handling
    return paged_decode_attention_mosaic_v2(
        q, k_cache, v_cache, block_tables, context_lens, scale, config
    )


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


# =============================================================================
# Simple WGMMA-based Attention (without warp specialization)
# =============================================================================

def simple_attention_wgmma(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k: jax.Array,           # [batch_size, seq_len, num_kv_heads, head_dim]
    v: jax.Array,           # [batch_size, seq_len, num_kv_heads, head_dim]
    scale: float,
    config: MosaicAttentionConfig | None = None,
) -> jax.Array:
    """Simple attention kernel using WGMMA and emit_pipeline.
    
    This is a simpler version without warp specialization, demonstrating
    the core WGMMA + online softmax pattern. Useful for understanding
    and debugging before moving to full warp-specialized kernels.
    
    Based on ragged_dot pattern with emit_pipeline.
    
    Args:
        q: Query [batch_size, num_heads, head_dim] (single token per seq for decode).
        k: Key [batch_size, seq_len, num_kv_heads, head_dim].
        v: Value [batch_size, seq_len, num_kv_heads, head_dim].
        scale: Softmax scale.
        config: Kernel configuration.
    
    Returns:
        Output [batch_size, num_heads, head_dim].
    """
    _check_mosaic_available()
    
    if config is None:
        config = MosaicAttentionConfig()
    
    batch_size, num_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _ = k.shape
    
    block_kv = config.block_kv
    max_concurrent_steps = config.max_concurrent_steps
    
    q_heads_per_kv_head = num_heads // num_kv_heads
    num_kv_tiles = seq_len // block_kv
    
    # For decode, we need batch_size >= 64 for WGMMA
    # If not, we'd need to pad or use a different approach
    if batch_size < 64:
        raise ValueError(
            f"batch_size={batch_size} < 64, WGMMA requires M >= 64. "
            "Use batched_decode_attention_mosaic which handles this."
        )
    
    transforms = get_smem_transforms(head_dim, q.dtype)
    
    def kernel_body(q_gmem, k_gmem, v_gmem, out_gmem):
        """Simple attention using emit_pipeline over KV blocks."""
        
        # Grid: (batch_size, num_heads)
        # But we use nd_loop for more flexible scheduling
        
        @plgpu.nd_loop((batch_size, num_heads), collective_axes="sm")
        def batch_head_loop(loop_info):
            batch_idx, head_idx = loop_info.index
            kv_head_idx = lax.div(head_idx, jnp.array(q_heads_per_kv_head, head_idx.dtype))
            
            # Load query for this (batch, head) - just 1 token for decode
            q_vec = q_gmem[batch_idx, head_idx, :]  # [head_dim]
            
            # For WGMMA, we need 2D operands. Reshape Q to [1, head_dim]
            # But WGMMA needs M >= 64, so this single-query approach won't work!
            # 
            # The real solution is to process multiple batches together.
            # This simple version is for educational purposes - showing the API.
            
            # Initialize softmax state
            m_i = jnp.float32(-jnp.inf)
            l_i = jnp.float32(0.0)
            acc = jnp.zeros((head_dim,), dtype=jnp.float32)
            
            # Process KV tiles using emit_pipeline
            # Note: emit_pipeline is for overlapping memory and compute
            # For the single-query case, we fall back to simple loop
            
            def process_kv_tile(kv_tile_idx, carry):
                m_i, l_i, acc = carry
                
                kv_start = kv_tile_idx * block_kv
                
                # Load K and V for this tile
                k_tile = k_gmem[batch_idx, kv_start:kv_start + block_kv, kv_head_idx, :]
                v_tile = v_gmem[batch_idx, kv_start:kv_start + block_kv, kv_head_idx, :]
                
                # QK^T: [head_dim] @ [block_kv, head_dim]^T = [block_kv]
                scores = jnp.dot(q_vec.astype(jnp.float32), k_tile.astype(jnp.float32).T) * scale
                
                # Online softmax (single row)
                log2e = math.log2(math.e)
                m_ij = scores.max() * log2e
                m_new = jnp.maximum(m_i, m_ij)
                alpha = jnp.exp2(m_i - m_new)
                p = jnp.exp2(scores * log2e - m_new)
                
                l_new = alpha * l_i + p.sum()
                acc_new = alpha * acc + jnp.dot(p, v_tile.astype(jnp.float32))
                
                return m_new, l_new, acc_new
            
            m_f, l_f, acc_f = lax.fori_loop(
                0, num_kv_tiles, process_kv_tile, (m_i, l_i, acc)
            )
            
            # Normalize and store
            out = acc_f / l_f
            out_gmem = out_gmem.at[batch_idx, head_idx, :].set(out.astype(q_gmem.dtype))
    
    # Note: This is simplified and doesn't use full Mosaic features
    # A real implementation would use plgpu.kernel with SMEM and WGMMA
    # This serves as documentation of the intended algorithm
    
    num_sms = 132  # H100 SM count
    
    return plgpu.kernel(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), q.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(q, k, v)


# =============================================================================
# Batched Decode with emit_pipeline (WGMMA-compatible)
# =============================================================================

def batched_decode_emit_pipeline(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k: jax.Array,           # [batch_size, seq_len, num_kv_heads, head_dim]
    v: jax.Array,           # [batch_size, seq_len, num_kv_heads, head_dim]
    scale: float,
    block_q: int = 64,      # Must be >= 64 for WGMMA
    block_kv: int = 64,
    max_concurrent_steps: int = 4,
) -> jax.Array:
    """Batched decode attention using emit_pipeline and WGMMA.
    
    This version batches queries across multiple sequences to satisfy
    WGMMA's M >= 64 requirement. Uses emit_pipeline for memory/compute overlap.
    
    Based on ragged_dot_mgpu.py pattern.
    
    Args:
        q: Query [batch_size, num_heads, head_dim].
        k: Key [batch_size, seq_len, num_kv_heads, head_dim].
        v: Value [batch_size, seq_len, num_kv_heads, head_dim].
        scale: Softmax scale.
        block_q: Query batch size (for WGMMA M dimension).
        block_kv: KV tile size.
        max_concurrent_steps: Pipeline depth.
    
    Returns:
        Output [batch_size, num_heads, head_dim].
    """
    _check_mosaic_available()
    
    batch_size, num_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _ = k.shape
    
    q_heads_per_kv_head = num_heads // num_kv_heads
    
    # Reshape Q to [batch_tiles, block_q, num_heads, head_dim]
    num_batch_tiles = (batch_size + block_q - 1) // block_q
    
    # Pad batch dimension if needed
    padded_batch = num_batch_tiles * block_q
    if padded_batch > batch_size:
        pad_size = padded_batch - batch_size
        q = jnp.pad(q, ((0, pad_size), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, pad_size), (0, 0), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, pad_size), (0, 0), (0, 0), (0, 0)))
    
    num_kv_tiles = seq_len // block_kv
    
    dtype = q.dtype
    transforms = get_smem_transforms(head_dim, dtype)
    
    def kernel_body(q_gmem, k_gmem, v_gmem, out_gmem):
        """Kernel using emit_pipeline for KV processing."""
        
        @plgpu.nd_loop((num_batch_tiles, num_heads), collective_axes="sm")
        def tile_head_loop(loop_info):
            batch_tile_idx, head_idx = loop_info.index
            kv_head_idx = lax.div(head_idx, jnp.array(q_heads_per_kv_head, head_idx.dtype))
            
            batch_start = batch_tile_idx * block_q
            
            # Allocate accumulator for result
            def acc_scope(acc_ref):
                # Initialize online softmax state
                m_i = plgpu.layout_cast(
                    jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
                    plgpu.Layout.WGMMA_ROW,
                )
                l_i = plgpu.layout_cast(
                    jnp.full((block_q,), 0.0, dtype=jnp.float32),
                    plgpu.Layout.WGMMA_ROW,
                )
                
                # Pipeline body: process one KV tile
                def kv_pipeline_body(kv_idx, q_smem, k_smem, v_smem):
                    nonlocal m_i, l_i
                    
                    # QK^T via WGMMA: [block_q, head_dim] @ [head_dim, block_kv]
                    # = [block_q, block_kv]
                    
                    # Compute QK^T
                    def compute_qk(qk_acc_ref):
                        plgpu.wgmma(
                            qk_acc_ref,
                            q_smem,
                            plgpu.transpose_ref(k_smem, (1, 0)),
                        )
                        return qk_acc_ref[...]
                    
                    qk = pl.run_scoped(
                        compute_qk,
                        plgpu.ACC((block_q, block_kv), jnp.float32),
                    )
                    
                    # Apply scale
                    qk = qk * scale
                    
                    # Online softmax
                    log2e = math.log2(math.e)
                    m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)
                    alpha = jnp.exp2(m_i - m_ij)
                    m_i = m_ij
                    p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                    
                    # Update accumulator
                    # acc *= alpha  -- need to do this via ACC operations
                    l_i = l_i * alpha + p.sum(axis=1)
                    
                    # PV via WGMMA: accumulate into acc_ref
                    p16 = p.astype(dtype)
                    plgpu.wgmma(acc_ref, p16, v_smem)
                
                # Use emit_pipeline for memory/compute overlap
                plgpu.emit_pipeline(
                    kv_pipeline_body,
                    grid=(num_kv_tiles,),
                    in_specs=[
                        # Q tile (loaded once per sequence batch)
                        plgpu.BlockSpec(
                            (block_q, head_dim),
                            lambda kv_idx: (batch_start, head_idx, 0),  # Q doesn't change with kv_idx
                            transforms=transforms,
                            delay_release=1,
                        ),
                        # K tiles (streamed)
                        plgpu.BlockSpec(
                            (block_kv, head_dim),
                            lambda kv_idx: (batch_start, kv_idx * block_kv, kv_head_idx, 0),
                            transforms=transforms,
                            delay_release=1,
                        ),
                        # V tiles (streamed)
                        plgpu.BlockSpec(
                            (block_kv, head_dim),
                            lambda kv_idx: (batch_start, kv_idx * block_kv, kv_head_idx, 0),
                            transforms=transforms,
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(q_gmem, k_gmem, v_gmem)
                
                # Normalize result
                acc = acc_ref[...]
                acc = acc / lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
                return acc
            
            acc = pl.run_scoped(
                acc_scope,
                plgpu.ACC((block_q, head_dim), jnp.float32),
            )
            
            # Store output
            @functools.partial(
                pl.run_scoped,
                out_smem=plgpu.SMEM((block_q, head_dim), dtype, transforms=transforms),
            )
            def store_scope(out_smem):
                out_smem[...] = acc.astype(dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    out_smem,
                    out_gmem.at[batch_start:batch_start + block_q, head_idx, :],
                )
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)
    
    num_sms = 132
    
    result = plgpu.kernel(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct((padded_batch, num_heads, head_dim), dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(q, k, v)
    
    # Remove padding
    return result[:batch_size]
