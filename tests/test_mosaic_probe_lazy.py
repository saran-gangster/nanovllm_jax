"""Tests for lazy Mosaic decode probe behavior."""

from __future__ import annotations

import nanovllm_jax.layers.paged_attention as pa


def test_mosaic_probe_is_lazy_for_small_batches(monkeypatch) -> None:
    calls = {"n": 0}
    state = pa.create_attention_backend_runtime_state()
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    pa.set_attention_backend_runtime_state(state)

    def fake_probe() -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", object())
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 512)
    state.probe_ok = False
    state.probe_attempted = False
    state.baseline_disabled_reason = None
    monkeypatch.setattr(pa, "_probe_mosaic_decode_startup", fake_probe)

    # Below threshold: should not probe.
    assert pa._ensure_mosaic_decode_probe_ready(batch_size=64, block_q=64) is False
    assert calls["n"] == 0
    assert state.probe_attempted is False

    # At threshold: probe should run exactly once.
    assert pa._ensure_mosaic_decode_probe_ready(batch_size=512, block_q=64) is True
    assert calls["n"] == 1
    assert state.probe_attempted is True

    # Already probed+ok: no repeat probes.
    assert pa._ensure_mosaic_decode_probe_ready(batch_size=512, block_q=64) is True
    assert calls["n"] == 1


def test_mosaic_probe_failure_is_not_retried(monkeypatch) -> None:
    calls = {"n": 0}
    state = pa.create_attention_backend_runtime_state()
    state.use_mosaic_paged_decode = True
    state.use_blockwise_decode = False
    pa.set_attention_backend_runtime_state(state)

    def fake_probe() -> bool:
        calls["n"] += 1
        return False

    monkeypatch.setattr(pa, "MOSAIC_AVAILABLE", True)
    monkeypatch.setattr(pa, "mosaic_attn", object())
    monkeypatch.setattr(pa, "_MOSAIC_MIN_DECODE_BATCH", 512)
    state.probe_ok = False
    state.probe_attempted = False
    state.baseline_disabled_reason = "probe failed"
    monkeypatch.setattr(pa, "_probe_mosaic_decode_startup", fake_probe)

    assert pa._ensure_mosaic_decode_probe_ready(batch_size=512, block_q=64) is False
    assert calls["n"] == 1
    assert state.probe_attempted is True

    # Failed probe should not be retried.
    assert pa._ensure_mosaic_decode_probe_ready(batch_size=512, block_q=64) is False
    assert calls["n"] == 1
