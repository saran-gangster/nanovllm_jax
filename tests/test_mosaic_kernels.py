"""Numerical regression tests for paged attention kernels."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nanovllm_jax.layers.paged_attention as pa
from nanovllm_jax.layers.attention import variable_length_attention_prefill


def _decode_inputs(dtype=jnp.float32):
    batch_size = 4
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    num_blocks = 16
    block_size = 64
    max_blocks_per_seq = 4

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    q = jax.random.normal(keys[0], (batch_size, num_heads, head_dim), dtype=dtype)
    k_cache = jax.random.normal(
        keys[1], (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype,
    )
    v_cache = jax.random.normal(
        keys[2], (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype,
    )
    block_tables = jnp.asarray(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=jnp.int32,
    )
    context_lens = jnp.asarray([63, 95, 127, 191], dtype=jnp.int32)
    scale = float(1.0 / math.sqrt(head_dim))
    return q, k_cache, v_cache, block_tables, context_lens, scale, block_size


def _prefill_inputs(dtype=jnp.float32):
    seq_lens = [5, 7, 4]
    batch_size = len(seq_lens)
    num_heads = 8
    num_kv_heads = 2
    head_dim = 32
    total_tokens = sum(seq_lens)
    cu_seqlens = jnp.asarray(
        [0] + [sum(seq_lens[: i + 1]) for i in range(batch_size)],
        dtype=jnp.int32,
    )
    max_seqlen = max(seq_lens)
    scale = float(1.0 / math.sqrt(head_dim))

    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 3)
    q = jax.random.normal(keys[0], (total_tokens, num_heads, head_dim), dtype=dtype)
    k = jax.random.normal(keys[1], (total_tokens, num_kv_heads, head_dim), dtype=dtype)
    v = jax.random.normal(keys[2], (total_tokens, num_kv_heads, head_dim), dtype=dtype)
    return q, k, v, cu_seqlens, max_seqlen, scale, batch_size, num_heads, num_kv_heads


def test_blockwise_decode_matches_vectorized_reference() -> None:
    q, k_cache, v_cache, block_tables, context_lens, scale, block_size = _decode_inputs()

    out_blockwise = pa.paged_decode_attention_blockwise(
        q, k_cache, v_cache, block_tables, context_lens, scale, block_size,
    )
    out_vectorized = pa.paged_decode_attention_vectorized(
        q, k_cache, v_cache, block_tables, context_lens, scale, block_size,
    )

    np.testing.assert_allclose(
        np.asarray(out_blockwise),
        np.asarray(out_vectorized),
        rtol=5e-5,
        atol=5e-5,
    )


def test_paged_attention_dispatch_matches_blockwise_path(monkeypatch) -> None:
    q, k_cache, v_cache, block_tables, context_lens, scale, block_size = _decode_inputs()
    state = pa.create_attention_backend_runtime_state()
    state.use_mosaic_paged_decode = False
    state.use_blockwise_decode = True
    pa.set_attention_backend_runtime_state(state)

    if hasattr(pa.paged_attention, "clear_cache"):
        pa.paged_attention.clear_cache()

    out_dispatch = pa.paged_attention(
        q, k_cache, v_cache, block_tables, context_lens, scale, block_size,
    )
    out_blockwise = pa.paged_decode_attention_blockwise(
        q, k_cache, v_cache, block_tables, context_lens, scale, block_size,
    )

    np.testing.assert_allclose(
        np.asarray(out_dispatch),
        np.asarray(out_blockwise),
        rtol=5e-5,
        atol=5e-5,
    )


def test_paged_prefill_matches_variable_length_reference(monkeypatch) -> None:
    q, k, v, cu_seqlens, max_seqlen, scale, batch_size, num_heads, num_kv_heads = _prefill_inputs()
    state = pa.create_attention_backend_runtime_state()
    state.prefill_disabled_reason = "force-reference-path"
    pa.set_attention_backend_runtime_state(state)

    out = pa.paged_prefill_attention(q, k, v, cu_seqlens, max_seqlen, scale)
    ref = variable_length_attention_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
        batch_size=batch_size,
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_smem_budget_cap_returns_valid_pipeline_depth() -> None:
    pma = pytest.importorskip("nanovllm_jax.layers.mosaic_gpu_attention")

    configs = [
        {"block_q": 64, "block_kv": 64, "head_dim": 128, "requested": 2},
        {"block_q": 64, "block_kv": 64, "head_dim": 128, "requested": 4},
        {"block_q": 64, "block_kv": 128, "head_dim": 128, "requested": 2},
        {"block_q": 128, "block_kv": 64, "head_dim": 128, "requested": 2},
    ]

    for cfg in configs:
        capped = pma._cap_pipeline_depth(
            block_q=cfg["block_q"],
            block_kv=cfg["block_kv"],
            head_dim=cfg["head_dim"],
            dtype=jnp.float16,
            num_compute_wgs=2,
            requested_steps=cfg["requested"],
            metadata_bytes=4096,
        )
        assert 0 < capped <= cfg["requested"]


def test_per_sequence_block_tables_change_decode_outputs() -> None:
    q, k_cache, v_cache, block_tables, context_lens, scale, block_size = _decode_inputs()

    shared_block_tables = jnp.tile(block_tables[:1], (block_tables.shape[0], 1))

    out_unique = pa.paged_decode_attention_blockwise(
        q, k_cache, v_cache, block_tables, context_lens, scale, block_size,
    )
    out_shared = pa.paged_decode_attention_blockwise(
        q, k_cache, v_cache, shared_block_tables, context_lens, scale, block_size,
    )

    assert not np.allclose(np.asarray(out_unique), np.asarray(out_shared))
