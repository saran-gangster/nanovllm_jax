"""Decode WG selection tests for Mosaic kernel."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from nanovllm_jax.layers.mosaic_gpu_attention import _select_decode_num_compute_wgs
from nanovllm_jax.layers.mosaic_gpu_attention import prepare_decode_metadata


def test_decode_wg_selection_enables_two_wgs_for_128_tile_and_large_batch() -> None:
    assert _select_decode_num_compute_wgs(
        requested_num_compute_wgs=2,
        block_q=128,
        batch_size=128,
    ) == 2


def test_decode_wg_selection_falls_back_for_small_batch() -> None:
    assert _select_decode_num_compute_wgs(
        requested_num_compute_wgs=2,
        block_q=128,
        batch_size=64,
    ) == 1


def test_decode_wg_selection_falls_back_when_rows_per_wg_break_wgmma_constraint() -> None:
    assert _select_decode_num_compute_wgs(
        requested_num_compute_wgs=2,
        block_q=64,
        batch_size=512,
    ) == 1


def test_decode_metadata_flat_positions_match_block_and_offset_metadata() -> None:
    block_tables = jnp.asarray(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ],
        dtype=jnp.int32,
    )
    context_lens = jnp.asarray([120, 300, 10, 512], dtype=jnp.int32)
    metadata = prepare_decode_metadata(
        block_tables=block_tables,
        context_lens=context_lens,
        batch_size=4,
        block_q=2,
        block_size=256,
        block_kv=64,
    )

    block_indices = np.asarray(metadata.tile_chunk_block_indices)
    offsets = np.asarray(metadata.tile_chunk_offsets)
    flat_positions = np.asarray(metadata.tile_chunk_flat_positions)
    chunk_counts = np.asarray(metadata.tile_chunk_counts)

    for tile_idx, count in enumerate(chunk_counts.tolist()):
        valid = int(count)
        if valid == 0:
            continue
        expected = block_indices[tile_idx, :valid] * 256 + offsets[tile_idx, :valid]
        actual = flat_positions[tile_idx, :valid]
        assert np.array_equal(
            actual, expected
        ), f"tile {tile_idx}: flat positions must equal block_id*block_size + offset"
