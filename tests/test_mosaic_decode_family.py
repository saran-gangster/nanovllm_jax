"""Tests for Mosaic decode kernel-family dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import nanovllm_jax.layers.paged_attention as pa


def _dummy_inputs():
    q = jnp.zeros((64, 8, 128), dtype=jnp.float16)
    k_cache = jnp.zeros((1024, 256, 8, 128), dtype=jnp.float16)
    v_cache = jnp.zeros_like(k_cache)
    block_tables = jnp.zeros((64, 16), dtype=jnp.int32)
    context_lens = jnp.full((64,), 1024, dtype=jnp.int32)
    return q, k_cache, v_cache, block_tables, context_lens


def _dispatch_state() -> pa.AttentionBackendRuntimeState:
    return pa.get_attention_backend_runtime_state()


@pytest.fixture(autouse=True)
def _reset_dispatch_state(monkeypatch) -> None:
    state = pa.create_attention_backend_runtime_state()
    state.use_mosaic_paged_decode = False
    state.use_blockwise_decode = True
    pa.set_attention_backend_runtime_state(state)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_FAMILY_TABLE", {})


def test_configure_attention_backends_sets_public_backend_only() -> None:
    cfg = SimpleNamespace(decode_attention_backend="mosaic")
    state = pa.create_attention_backend_runtime_state()

    pa._MOSAIC_DECODE_KERNEL_FAMILY = "latency"
    pa.configure_attention_backends(cfg, runtime_state=state)

    assert state.use_mosaic_paged_decode is True
    assert state.use_blockwise_decode is False
    assert pa._MOSAIC_DECODE_KERNEL_FAMILY == "latency"


def test_configure_attention_backends_rejects_unknown_backend() -> None:
    cfg = SimpleNamespace(decode_attention_backend="unknown")
    with pytest.raises(ValueError, match="auto\\|mosaic\\|blockwise"):
        pa.configure_attention_backends(cfg)


def test_select_mosaic_decode_variant_auto_policy() -> None:
    pa._MOSAIC_DECODE_FAMILY_TABLE = {}
    _dispatch_state().variant_selection_cache.clear()

    assert pa._select_mosaic_decode_variant(
        requested_variant="auto",
        padded_batch=512,
        head_dim=128,
        max_blocks_per_seq=32,
        block_size=256,
    ) == "throughput"
    assert pa._select_mosaic_decode_variant(
        requested_variant="auto",
        padded_batch=512,
        head_dim=128,
        max_blocks_per_seq=24,
        block_size=256,
    ) == "baseline"
    assert pa._select_mosaic_decode_variant(
        requested_variant="auto",
        padded_batch=128,
        head_dim=128,
        max_blocks_per_seq=8,
        block_size=256,
    ) == "latency"


def test_select_mosaic_decode_variant_table_override_and_fallback() -> None:
    key = (512, 128, 16, 256)
    pa._MOSAIC_DECODE_FAMILY_TABLE = {key: "baseline"}
    _dispatch_state().variant_selection_cache.clear()

    selected = pa._select_mosaic_decode_variant(
        requested_variant="auto",
        padded_batch=512,
        head_dim=128,
        max_blocks_per_seq=16,
        block_size=256,
    )
    assert selected == "baseline"

    selected_missing = pa._select_mosaic_decode_variant(
        requested_variant="auto",
        padded_batch=512,
        head_dim=128,
        max_blocks_per_seq=48,
        block_size=256,
    )
    assert selected_missing == "throughput"


def test_mosaic_decode_prefers_latency_family(monkeypatch) -> None:
    latency_out = object()
    state = _dispatch_state()

    class DummyMosaic:
        @staticmethod
        def paged_decode_attention_mosaic_latency(**_kwargs):
            return latency_out

        @staticmethod
        def MosaicAttentionConfig(**_kwargs):
            return SimpleNamespace()

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", DummyMosaic)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_Q", 64)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_KV", 256)
    monkeypatch.setattr(pa, "_MOSAIC_MAX_CONCURRENT_STEPS", 2)
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_KERNEL_FAMILY", "latency")
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    state.baseline_disabled_reason = None
    state.latency_disabled_reason = None

    def _fail_probe(**_kwargs):
        raise AssertionError("baseline probe should not run for latency family")

    monkeypatch.setattr(pa, "_ensure_mosaic_decode_probe_ready", _fail_probe)

    q, k_cache, v_cache, block_tables, context_lens = _dummy_inputs()
    out = pa._maybe_run_mosaic_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0,
        block_size=256,
    )
    assert out is latency_out


def test_mosaic_decode_prefers_throughput_family(monkeypatch) -> None:
    throughput_out = object()
    state = _dispatch_state()

    class DummyMosaic:
        @staticmethod
        def paged_decode_attention_mosaic_throughput(**_kwargs):
            return throughput_out

        @staticmethod
        def MosaicAttentionConfig(**_kwargs):
            return SimpleNamespace()

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", DummyMosaic)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_Q", 64)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_KV", 256)
    monkeypatch.setattr(pa, "_MOSAIC_MAX_CONCURRENT_STEPS", 2)
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_KERNEL_FAMILY", "throughput")
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_SPLIT_K", 2)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_NUM_STAGES", 2)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_RESCALE_THRESHOLD", 1.0)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_AUTOTUNE", False)
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    state.baseline_disabled_reason = None
    state.latency_disabled_reason = None
    state.throughput_disabled_reason = None

    def _fail_probe(**_kwargs):
        raise AssertionError("baseline probe should not run for throughput family")

    monkeypatch.setattr(pa, "_ensure_mosaic_decode_probe_ready", _fail_probe)

    q, k_cache, v_cache, block_tables, context_lens = _dummy_inputs()
    out = pa._maybe_run_mosaic_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0,
        block_size=256,
    )
    assert out is throughput_out


def test_latency_failure_falls_back_to_baseline(monkeypatch) -> None:
    baseline_out = object()
    state = _dispatch_state()

    class DummyMosaic:
        @staticmethod
        def paged_decode_attention_mosaic_latency(**_kwargs):
            raise RuntimeError("latency kernel failed")

        @staticmethod
        def MosaicAttentionConfig(**_kwargs):
            return SimpleNamespace()

        @staticmethod
        def prepare_decode_metadata(*_args, **_kwargs):
            return SimpleNamespace()

        @staticmethod
        def batched_decode_attention_mosaic(**_kwargs):
            return baseline_out

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", DummyMosaic)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_Q", 64)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_KV", 256)
    monkeypatch.setattr(pa, "_MOSAIC_MAX_CONCURRENT_STEPS", 2)
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_KERNEL_FAMILY", "latency")
    monkeypatch.setattr(pa, "_ensure_mosaic_decode_probe_ready", lambda **_kwargs: True)
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    state.baseline_disabled_reason = None
    state.latency_disabled_reason = None
    state.tile_cache = {}

    q, k_cache, v_cache, block_tables, context_lens = _dummy_inputs()
    out = pa._maybe_run_mosaic_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0,
        block_size=256,
    )
    assert out is baseline_out
    assert state.latency_disabled_reason is not None


def test_throughput_failure_falls_back_to_baseline(monkeypatch) -> None:
    baseline_out = object()
    state = _dispatch_state()

    class DummyMosaic:
        @staticmethod
        def paged_decode_attention_mosaic_throughput(**_kwargs):
            raise RuntimeError("throughput kernel failed")

        @staticmethod
        def MosaicAttentionConfig(**_kwargs):
            return SimpleNamespace()

        @staticmethod
        def prepare_decode_metadata(*_args, **_kwargs):
            return SimpleNamespace()

        @staticmethod
        def batched_decode_attention_mosaic(**_kwargs):
            return baseline_out

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", DummyMosaic)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_Q", 64)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_KV", 256)
    monkeypatch.setattr(pa, "_MOSAIC_MAX_CONCURRENT_STEPS", 2)
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_KERNEL_FAMILY", "throughput")
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_SPLIT_K", 2)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_NUM_STAGES", 2)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_RESCALE_THRESHOLD", 1.0)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_AUTOTUNE", False)
    monkeypatch.setattr(pa, "_ensure_mosaic_decode_probe_ready", lambda **_kwargs: True)
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    state.baseline_disabled_reason = None
    state.latency_disabled_reason = None
    state.throughput_disabled_reason = None
    state.tile_cache = {}

    q, k_cache, v_cache, block_tables, context_lens = _dummy_inputs()
    out = pa._maybe_run_mosaic_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0,
        block_size=256,
    )
    assert out is baseline_out
    assert state.throughput_disabled_reason is not None


def test_auto_throughput_falls_back_to_latency(monkeypatch) -> None:
    latency_out = object()
    state = _dispatch_state()

    class DummyMosaic:
        @staticmethod
        def paged_decode_attention_mosaic_throughput(**_kwargs):
            raise RuntimeError("throughput kernel failed")

        @staticmethod
        def paged_decode_attention_mosaic_latency(**_kwargs):
            return latency_out

        @staticmethod
        def MosaicAttentionConfig(**_kwargs):
            return SimpleNamespace()

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", DummyMosaic)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_Q", 64)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_KV", 256)
    monkeypatch.setattr(pa, "_MOSAIC_MAX_CONCURRENT_STEPS", 2)
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_KERNEL_FAMILY", "auto")
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_SPLIT_K", 2)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_NUM_STAGES", 2)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_RESCALE_THRESHOLD", 1.0)
    monkeypatch.setattr(pa, "_MOSAIC_THROUGHPUT_AUTOTUNE", False)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_FAMILY_TABLE", {})
    monkeypatch.setattr(pa, "_ensure_mosaic_decode_probe_ready", lambda **_kwargs: False)
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    state.baseline_disabled_reason = None
    state.latency_disabled_reason = None
    state.throughput_disabled_reason = None
    state.variant_selection_cache = {}

    q = jnp.zeros((512, 8, 128), dtype=jnp.float16)
    k_cache = jnp.zeros((1, 256, 8, 128), dtype=jnp.float16)
    v_cache = jnp.zeros_like(k_cache)
    block_tables = jnp.zeros((512, 32), dtype=jnp.int32)
    context_lens = jnp.full((512,), 1024, dtype=jnp.int32)

    out = pa._maybe_run_mosaic_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0,
        block_size=256,
    )
    assert out is latency_out
    assert state.throughput_disabled_reason is not None


def test_latency_family_uses_baseline_on_ineligible_shapes(monkeypatch) -> None:
    baseline_out = object()
    state = _dispatch_state()

    class DummyMosaic:
        @staticmethod
        def paged_decode_attention_mosaic_latency(**_kwargs):
            raise AssertionError("latency family should be skipped for ineligible shapes")

        @staticmethod
        def MosaicAttentionConfig(**_kwargs):
            return SimpleNamespace()

        @staticmethod
        def prepare_decode_metadata(*_args, **_kwargs):
            return SimpleNamespace()

        @staticmethod
        def batched_decode_attention_mosaic(**_kwargs):
            return baseline_out

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", DummyMosaic)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_Q", 64)
    monkeypatch.setattr(pa, "_MOSAIC_BLOCK_KV", 256)
    monkeypatch.setattr(pa, "_MOSAIC_MAX_CONCURRENT_STEPS", 2)
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 0)
    monkeypatch.setattr(pa, "_MOSAIC_DECODE_KERNEL_FAMILY", "latency")
    monkeypatch.setattr(pa, "_ensure_mosaic_decode_probe_ready", lambda **_kwargs: True)
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    state.baseline_disabled_reason = None
    state.latency_disabled_reason = None
    state.tile_cache = {}

    q = jnp.zeros((64, 8, 64), dtype=jnp.float16)
    k_cache = jnp.zeros((128, 256, 8, 64), dtype=jnp.float16)
    v_cache = jnp.zeros_like(k_cache)
    block_tables = jnp.zeros((64, 2), dtype=jnp.int32)
    context_lens = jnp.full((64,), 128, dtype=jnp.int32)

    out = pa._maybe_run_mosaic_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0,
        block_size=256,
    )
    assert out is baseline_out
