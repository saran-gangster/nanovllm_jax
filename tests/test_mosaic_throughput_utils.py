"""Unit tests for Mosaic throughput helper utilities."""

from __future__ import annotations

import jax.numpy as jnp

import nanovllm_jax.layers.mosaic_gpu_attention as mga


def test_merge_split_partials_matches_reference_formula() -> None:
    acc_partial = jnp.array(
        [[[[1.0, 2.0]], [[3.0, 4.0]]]],
        dtype=jnp.float32,
    )
    l_partial = jnp.array([[[2.0], [1.5]]], dtype=jnp.float32)
    m_partial = jnp.array([[[1.0], [0.5]]], dtype=jnp.float32)

    out = mga._merge_split_partials(acc_partial, l_partial, m_partial, axis=1)

    m_next = jnp.max(m_partial, axis=1)
    corr = jnp.exp2(m_partial - m_next[:, None, :])
    ref = jnp.sum(acc_partial * corr[..., None], axis=1) / jnp.sum(
        l_partial * corr, axis=1
    )[..., None]
    assert jnp.allclose(out, ref, atol=1e-6)


def test_select_throughput_k_splits_respects_divisibility(monkeypatch) -> None:
    monkeypatch.setattr(mga, "_default_core_count", lambda: 132)

    splits = mga._select_throughput_k_splits(
        split_k=7,
        batch_size=512,
        head_dim=128,
        num_heads=32,
        block_q=64,
        max_blocks_per_seq=16,
        block_size=256,
        block_kv=64,
    )

    num_kv_tiles = 16 * (256 // 64)
    assert splits == 4
    assert num_kv_tiles % splits == 0


def test_select_throughput_k_splits_heuristic_bounds(monkeypatch) -> None:
    monkeypatch.setattr(mga, "_default_core_count", lambda: 132)

    splits = mga._select_throughput_k_splits(
        split_k=0,
        batch_size=128,
        head_dim=128,
        num_heads=32,
        block_q=64,
        max_blocks_per_seq=48,
        block_size=256,
        block_kv=64,
    )

    num_kv_tiles = 48 * (256 // 64)
    assert 1 <= splits <= 8
    assert num_kv_tiles % splits == 0


def test_select_throughput_k_splits_table_override(monkeypatch) -> None:
    monkeypatch.setattr(
        mga,
        "_lookup_throughput_splitk_override",
        lambda **_kwargs: 6,
    )
    splits = mga._select_throughput_k_splits(
        split_k=0,
        batch_size=512,
        head_dim=128,
        num_heads=32,
        block_q=64,
        max_blocks_per_seq=16,
        block_size=256,
        block_kv=64,
    )
    assert splits == 4
