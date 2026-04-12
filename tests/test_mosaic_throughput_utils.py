"""Unit tests for Mosaic throughput helper utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import nanovllm_jax.layers.mosaic_gpu_attention as mga
from nanovllm_jax.utils.runtime_diagnostics import (
    consume_partitioned_decode_reduction_stats,
    reset_partitioned_decode_reduction_stats,
)


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


def test_reduce_partitioned_decode_partials_records_backend_and_family(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NANOVLLM_JAX_PROFILE_DECODE_STEP", "1")
    reset_partitioned_decode_reduction_stats()

    acc_partial = jnp.array(
        [[[[1.0, 2.0]], [[3.0, 4.0]]]],
        dtype=jnp.float32,
    )
    l_partial = jnp.array([[[2.0], [1.5]]], dtype=jnp.float32)
    m_partial = jnp.array([[[1.0], [0.5]]], dtype=jnp.float32)

    out = mga.reduce_partitioned_decode_partials(
        acc_partial,
        l_partial,
        m_partial,
        axis=1,
        family="throughput",
        backend_override="streaming",
    )
    stats = consume_partitioned_decode_reduction_stats()

    assert out.shape == (1, 1, 2)
    assert stats["calls"] == 1
    assert stats["backend"] == "streaming"
    assert stats["family"] == "throughput"
    assert stats["max_splits"] == 2
    assert stats["measured"] is True
    assert stats["seconds"] >= 0.0


def test_reduce_partitioned_decode_partials_rejects_unknown_backend() -> None:
    acc_partial = jnp.ones((1, 2, 1, 2), dtype=jnp.float32)
    l_partial = jnp.ones((1, 2, 1), dtype=jnp.float32)
    m_partial = jnp.ones((1, 2, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="auto\\|streaming"):
        mga.reduce_partitioned_decode_partials(
            acc_partial,
            l_partial,
            m_partial,
            axis=1,
            family="latency",
            backend_override="unknown",
        )


def test_build_throughput_v2_plan_reports_non_wrapper_target() -> None:
    q = jnp.zeros((2, 8, 128), dtype=jnp.float16)
    block_tables = jnp.zeros((2, 16), dtype=jnp.int32)
    context_lens = jnp.full((2,), 4096, dtype=jnp.int32)

    plan = mga.build_paged_decode_throughput_v2_plan(
        q=q,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=256,
        split_k=2,
    )

    assert plan.batch_size == 2
    assert plan.max_blocks_per_seq == 16
    assert plan.k_splits >= 1
    assert plan.pages_per_partition >= 1
    assert plan.uses_wrapper_partitioning is False
    assert plan.reduction_boundary == "jax_merge_v1_fallback"
    assert plan.metadata_cache_key[0] == "throughput_v2"


def test_throughput_partitioned_decode_uses_reduction_boundary(monkeypatch) -> None:
    monkeypatch.setattr(mga, "_check_mosaic_available", lambda: None)
    monkeypatch.setattr(
        mga,
        "_select_throughput_k_splits",
        lambda **_kwargs: 2,
    )

    recorded = {}

    def _fake_prepare_decode_metadata(*_args, **_kwargs):
        return object()

    def _fake_batched_decode_attention_mosaic(
        *,
        q,
        return_partials=False,
        **_kwargs,
    ):
        assert return_partials is True
        batch, heads, dim = q.shape
        acc = jnp.ones((batch, heads, dim), dtype=jnp.float32)
        l = jnp.ones((batch, heads), dtype=jnp.float32)
        m = jnp.zeros((batch, heads), dtype=jnp.float32)
        return acc, l, m

    def _fake_reduce_partitioned_decode_partials(
        acc_partial,
        l_partial,
        m_partial,
        *,
        axis,
        family,
        backend_override=None,
    ):
        del l_partial, m_partial, backend_override
        recorded["shape"] = acc_partial.shape
        recorded["axis"] = axis
        recorded["family"] = family
        return jnp.full(
            (acc_partial.shape[0], acc_partial.shape[2], acc_partial.shape[3]),
            7.0,
            dtype=jnp.float32,
        )

    monkeypatch.setattr(mga, "prepare_decode_metadata", _fake_prepare_decode_metadata)
    monkeypatch.setattr(
        mga,
        "batched_decode_attention_mosaic",
        _fake_batched_decode_attention_mosaic,
    )
    monkeypatch.setattr(
        mga,
        "reduce_partitioned_decode_partials",
        _fake_reduce_partitioned_decode_partials,
    )

    q = jnp.zeros((2, 8, 128), dtype=jnp.float16)
    k_cache = jnp.zeros((64, 256, 8, 128), dtype=jnp.float16)
    v_cache = jnp.zeros_like(k_cache)
    block_tables = jnp.zeros((2, 16), dtype=jnp.int32)
    context_lens = jnp.full((2,), 4096, dtype=jnp.int32)

    out = mga.paged_decode_attention_mosaic_throughput(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0,
        block_size=256,
    )

    assert recorded == {
        "shape": (2, 2, 8, 128),
        "axis": 1,
        "family": "throughput",
    }
    assert out.shape == q.shape
    assert out.dtype == q.dtype


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
