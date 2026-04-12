"""Tests for runtime diagnostics counters."""

from __future__ import annotations

import pytest

from nanovllm_jax.utils.runtime_diagnostics import (
    consume_kv_update_stats,
    record_kv_update_stats,
    reset_kv_update_stats,
)


def test_kv_update_stats_accumulate_and_reset() -> None:
    reset_kv_update_stats()

    record_kv_update_stats(
        seconds=0.01,
        tokens=4,
        valid_tokens=3,
        skipped_tokens=1,
        duplicate_slots=1,
        backend="scatter",
        measured=False,
    )
    record_kv_update_stats(
        seconds=0.02,
        tokens=6,
        valid_tokens=6,
        skipped_tokens=0,
        duplicate_slots=0,
        backend="sorted_compact_scatter",
        measured=True,
    )

    snapshot = consume_kv_update_stats()
    assert snapshot["seconds"] == pytest.approx(0.03)
    assert snapshot["calls"] == 2
    assert snapshot["tokens"] == 10
    assert snapshot["valid_tokens"] == 9
    assert snapshot["skipped_tokens"] == 1
    assert snapshot["duplicate_slots"] == 1
    assert snapshot["backend"] == "sorted_compact_scatter"
    assert snapshot["measured"] is True

    reset_snapshot = consume_kv_update_stats()
    assert reset_snapshot == {
        "seconds": 0.0,
        "calls": 0,
        "tokens": 0,
        "valid_tokens": 0,
        "skipped_tokens": 0,
        "duplicate_slots": 0,
        "backend": None,
        "measured": False,
    }
