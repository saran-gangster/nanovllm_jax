"""Tests for the internal KV-cache update operator boundary."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import nanovllm_jax.layers.attention as attn

from nanovllm_jax.layers.attention import update_kv_cache, update_kv_cache_int8


def test_update_kv_cache_matches_scatter_semantics() -> None:
    key = jnp.array(
        [
            [[1.0, 2.0]],
            [[9.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [[3.0, 4.0]],
            [[11.0, 12.0]],
        ],
        dtype=jnp.float32,
    )
    k_cache = jnp.zeros((4, 1, 2), dtype=jnp.bfloat16)
    v_cache = jnp.zeros((4, 1, 2), dtype=jnp.bfloat16)
    slot_mapping = jnp.array([1, -1], dtype=jnp.int32)

    k_out, v_out = update_kv_cache(key, value, k_cache, v_cache, slot_mapping)

    k_np = np.asarray(k_out)
    v_np = np.asarray(v_out)
    assert np.allclose(k_np[1], np.asarray(key[0], dtype=np.float32), atol=0.01)
    assert np.allclose(v_np[1], np.asarray(value[0], dtype=np.float32), atol=0.01)
    assert np.allclose(k_np[0], 0.0)
    assert np.allclose(k_np[2], 0.0)
    assert np.allclose(k_np[3], 0.0)


def test_update_kv_cache_int8_skips_invalid_slots() -> None:
    key = jnp.array(
        [
            [[1.0, -1.0]],
            [[100.0, -100.0]],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [[0.5, -0.5]],
            [[50.0, -50.0]],
        ],
        dtype=jnp.float32,
    )
    k_cache = jnp.zeros((4, 1, 2), dtype=jnp.int8)
    v_cache = jnp.zeros((4, 1, 2), dtype=jnp.int8)
    k_scale = jnp.zeros((4, 1), dtype=jnp.float16)
    v_scale = jnp.zeros((4, 1), dtype=jnp.float16)
    slot_mapping = jnp.array([2, -1], dtype=jnp.int32)

    k_out, v_out, k_scale_out, v_scale_out = update_kv_cache_int8(
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
    )

    assert np.any(np.asarray(k_out[2]) != 0)
    assert np.any(np.asarray(v_out[2]) != 0)
    assert np.any(np.asarray(k_scale_out[2]) != 0)
    assert np.any(np.asarray(v_scale_out[2]) != 0)
    assert np.all(np.asarray(k_out[0]) == 0)
    assert np.all(np.asarray(k_out[1]) == 0)
    assert np.all(np.asarray(k_out[3]) == 0)


def test_update_kv_cache_compact_scatter_matches_reference(monkeypatch) -> None:
    key = jnp.array(
        [
            [[1.0, 2.0]],
            [[5.0, 6.0]],
            [[9.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [[3.0, 4.0]],
            [[7.0, 8.0]],
            [[11.0, 12.0]],
        ],
        dtype=jnp.float32,
    )
    k_cache = jnp.zeros((5, 1, 2), dtype=jnp.bfloat16)
    v_cache = jnp.zeros((5, 1, 2), dtype=jnp.bfloat16)
    slot_mapping = jnp.array([1, -1, 3], dtype=jnp.int32)

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "scatter")
    ref_k, ref_v = update_kv_cache(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        slot_mapping,
    )

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "compact_scatter")
    alt_k, alt_v = update_kv_cache(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        slot_mapping,
    )

    assert np.allclose(np.asarray(alt_k), np.asarray(ref_k), atol=0.01)
    assert np.allclose(np.asarray(alt_v), np.asarray(ref_v), atol=0.01)


def test_update_kv_cache_int8_compact_scatter_matches_reference(monkeypatch) -> None:
    key = jnp.array(
        [
            [[1.0, -1.0]],
            [[5.0, -5.0]],
            [[9.0, -9.0]],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [[0.5, -0.5]],
            [[2.5, -2.5]],
            [[4.5, -4.5]],
        ],
        dtype=jnp.float32,
    )
    k_cache = jnp.zeros((5, 1, 2), dtype=jnp.int8)
    v_cache = jnp.zeros((5, 1, 2), dtype=jnp.int8)
    k_scale = jnp.zeros((5, 1), dtype=jnp.float16)
    v_scale = jnp.zeros((5, 1), dtype=jnp.float16)
    slot_mapping = jnp.array([2, -1, 4], dtype=jnp.int32)

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "scatter")
    ref = update_kv_cache_int8(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1), dtype=jnp.float16),
        jnp.zeros((5, 1), dtype=jnp.float16),
        slot_mapping,
    )

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "compact_scatter")
    alt = update_kv_cache_int8(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1), dtype=jnp.float16),
        jnp.zeros((5, 1), dtype=jnp.float16),
        slot_mapping,
    )

    for alt_arr, ref_arr in zip(alt, ref, strict=True):
        assert np.allclose(np.asarray(alt_arr), np.asarray(ref_arr), atol=0.01)


def test_update_kv_cache_sorted_compact_scatter_matches_reference(monkeypatch) -> None:
    key = jnp.array(
        [
            [[1.0, 2.0]],
            [[5.0, 6.0]],
            [[9.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [[3.0, 4.0]],
            [[7.0, 8.0]],
            [[11.0, 12.0]],
        ],
        dtype=jnp.float32,
    )
    slot_mapping = jnp.array([3, -1, 1], dtype=jnp.int32)

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "scatter")
    ref_k, ref_v = update_kv_cache(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        slot_mapping,
    )

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "sorted_compact_scatter")
    alt_k, alt_v = update_kv_cache(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        slot_mapping,
    )

    assert np.allclose(np.asarray(alt_k), np.asarray(ref_k), atol=0.01)
    assert np.allclose(np.asarray(alt_v), np.asarray(ref_v), atol=0.01)


def test_update_kv_cache_int8_sorted_compact_scatter_matches_reference(monkeypatch) -> None:
    key = jnp.array(
        [
            [[1.0, -1.0]],
            [[5.0, -5.0]],
            [[9.0, -9.0]],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [[0.5, -0.5]],
            [[2.5, -2.5]],
            [[4.5, -4.5]],
        ],
        dtype=jnp.float32,
    )
    slot_mapping = jnp.array([4, -1, 2], dtype=jnp.int32)

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "scatter")
    ref = update_kv_cache_int8(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1), dtype=jnp.float16),
        jnp.zeros((5, 1), dtype=jnp.float16),
        slot_mapping,
    )

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "sorted_compact_scatter")
    alt = update_kv_cache_int8(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1, 2), dtype=jnp.int8),
        jnp.zeros((5, 1), dtype=jnp.float16),
        jnp.zeros((5, 1), dtype=jnp.float16),
        slot_mapping,
    )

    for alt_arr, ref_arr in zip(alt, ref, strict=True):
        assert np.allclose(np.asarray(alt_arr), np.asarray(ref_arr), atol=0.01)


def test_update_kv_cache_sorted_compact_scatter_preserves_duplicate_slot_result(
    monkeypatch,
) -> None:
    key = jnp.array(
        [
            [[1.0, 2.0]],
            [[5.0, 6.0]],
            [[9.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [[3.0, 4.0]],
            [[7.0, 8.0]],
            [[11.0, 12.0]],
        ],
        dtype=jnp.float32,
    )
    slot_mapping = jnp.array([2, 2, -1], dtype=jnp.int32)

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "scatter")
    ref_k, ref_v = update_kv_cache(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        slot_mapping,
    )

    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "sorted_compact_scatter")
    alt_k, alt_v = update_kv_cache(
        key,
        value,
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        jnp.zeros((5, 1, 2), dtype=jnp.bfloat16),
        slot_mapping,
    )

    assert np.allclose(np.asarray(alt_k), np.asarray(ref_k), atol=0.01)
    assert np.allclose(np.asarray(alt_v), np.asarray(ref_v), atol=0.01)


def test_update_kv_cache_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "unknown")

    with pytest.raises(
        ValueError,
        match="auto\\|scatter\\|compact_scatter\\|sorted_compact_scatter",
    ):
        update_kv_cache(
            jnp.zeros((1, 1, 2), dtype=jnp.float32),
            jnp.zeros((1, 1, 2), dtype=jnp.float32),
            jnp.zeros((2, 1, 2), dtype=jnp.bfloat16),
            jnp.zeros((2, 1, 2), dtype=jnp.bfloat16),
            jnp.array([0], dtype=jnp.int32),
        )


def test_active_kv_update_backend_prefers_current_env_override(monkeypatch) -> None:
    monkeypatch.setattr(attn, "_KV_UPDATE_BACKEND", "scatter")
    monkeypatch.setenv("NANOVLLM_JAX_KV_UPDATE_BACKEND", "compact_scatter")
    assert attn._active_kv_update_backend() == "compact_scatter"

    monkeypatch.setenv("NANOVLLM_JAX_KV_UPDATE_BACKEND", "sorted_compact_scatter")
    assert attn._active_kv_update_backend() == "sorted_compact_scatter"

    monkeypatch.delenv("NANOVLLM_JAX_KV_UPDATE_BACKEND", raising=False)
    assert attn._active_kv_update_backend() == "scatter"
