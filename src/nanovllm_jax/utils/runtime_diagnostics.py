"""Internal runtime diagnostics for decode-step profiling and schedule dumps."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import jax


_PROFILE_DECODE_STEP_ENV = "NANOVLLM_JAX_PROFILE_DECODE_STEP"
_DUMP_DECODE_SCHEDULE_ENV = "NANOVLLM_JAX_DUMP_DECODE_SCHEDULE"
_DIAGNOSTICS_DIR_ENV = "NANOVLLM_JAX_DIAGNOSTICS_DIR"

_KV_UPDATE_STATS_LOCK = threading.Lock()
_KV_UPDATE_STATS = {
    "seconds": 0.0,
    "calls": 0,
    "tokens": 0,
    "valid_tokens": 0,
    "skipped_tokens": 0,
    "duplicate_slots": 0,
    "backend": None,
    "measured": False,
}
_PARTITIONED_DECODE_REDUCTION_STATS_LOCK = threading.Lock()
_PARTITIONED_DECODE_REDUCTION_STATS = {
    "seconds": 0.0,
    "calls": 0,
    "backend": None,
    "family": None,
    "max_splits": 0,
    "measured": False,
}


def decode_step_profiling_enabled() -> bool:
    return os.environ.get(_PROFILE_DECODE_STEP_ENV, "0") == "1"


def decode_schedule_dump_enabled() -> bool:
    return os.environ.get(_DUMP_DECODE_SCHEDULE_ENV, "0") == "1"


def diagnostics_dir() -> Path:
    root = os.environ.get(_DIAGNOSTICS_DIR_ENV)
    base = Path(root) if root else Path(tempfile.gettempdir()) / "nanovllm_jax_diagnostics"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def append_jsonl(filename: str, payload: dict[str, Any]) -> None:
    path = diagnostics_dir() / filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")


def append_decode_step_record(payload: dict[str, Any]) -> None:
    append_jsonl("decode_step_profile.jsonl", payload)


def append_decode_schedule_record(payload: dict[str, Any]) -> None:
    append_jsonl("decode_schedule.jsonl", payload)


def block_until_ready_tree(value):
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def reset_kv_update_stats() -> None:
    with _KV_UPDATE_STATS_LOCK:
        _KV_UPDATE_STATS.update(
            {
                "seconds": 0.0,
                "calls": 0,
                "tokens": 0,
                "valid_tokens": 0,
                "skipped_tokens": 0,
                "duplicate_slots": 0,
                "backend": None,
                "measured": False,
            }
        )


def record_kv_update_stats(
    *,
    seconds: float,
    tokens: int,
    valid_tokens: int,
    skipped_tokens: int,
    duplicate_slots: int,
    backend: str,
    measured: bool,
) -> None:
    with _KV_UPDATE_STATS_LOCK:
        _KV_UPDATE_STATS["seconds"] += float(seconds)
        _KV_UPDATE_STATS["calls"] += 1
        _KV_UPDATE_STATS["tokens"] += int(tokens)
        _KV_UPDATE_STATS["valid_tokens"] += int(valid_tokens)
        _KV_UPDATE_STATS["skipped_tokens"] += int(skipped_tokens)
        _KV_UPDATE_STATS["duplicate_slots"] += int(duplicate_slots)
        _KV_UPDATE_STATS["backend"] = backend
        _KV_UPDATE_STATS["measured"] = bool(_KV_UPDATE_STATS["measured"] or measured)


def consume_kv_update_stats() -> dict[str, Any]:
    with _KV_UPDATE_STATS_LOCK:
        snapshot = dict(_KV_UPDATE_STATS)
        _KV_UPDATE_STATS.update(
            {
                "seconds": 0.0,
                "calls": 0,
                "tokens": 0,
                "valid_tokens": 0,
                "skipped_tokens": 0,
                "duplicate_slots": 0,
                "backend": None,
                "measured": False,
            }
        )
    return snapshot


def reset_partitioned_decode_reduction_stats() -> None:
    with _PARTITIONED_DECODE_REDUCTION_STATS_LOCK:
        _PARTITIONED_DECODE_REDUCTION_STATS.update(
            {
                "seconds": 0.0,
                "calls": 0,
                "backend": None,
                "family": None,
                "max_splits": 0,
                "measured": False,
            }
        )


def record_partitioned_decode_reduction_stats(
    *,
    seconds: float,
    backend: str,
    family: str,
    splits: int,
    measured: bool,
) -> None:
    with _PARTITIONED_DECODE_REDUCTION_STATS_LOCK:
        _PARTITIONED_DECODE_REDUCTION_STATS["seconds"] += float(seconds)
        _PARTITIONED_DECODE_REDUCTION_STATS["calls"] += 1
        _PARTITIONED_DECODE_REDUCTION_STATS["backend"] = backend
        _PARTITIONED_DECODE_REDUCTION_STATS["family"] = family
        _PARTITIONED_DECODE_REDUCTION_STATS["max_splits"] = max(
            int(_PARTITIONED_DECODE_REDUCTION_STATS["max_splits"]),
            int(splits),
        )
        _PARTITIONED_DECODE_REDUCTION_STATS["measured"] = bool(
            _PARTITIONED_DECODE_REDUCTION_STATS["measured"] or measured
        )


def consume_partitioned_decode_reduction_stats() -> dict[str, Any]:
    with _PARTITIONED_DECODE_REDUCTION_STATS_LOCK:
        snapshot = dict(_PARTITIONED_DECODE_REDUCTION_STATS)
        _PARTITIONED_DECODE_REDUCTION_STATS.update(
            {
                "seconds": 0.0,
                "calls": 0,
                "backend": None,
                "family": None,
                "max_splits": 0,
                "measured": False,
            }
        )
    return snapshot
