"""Helpers for collecting stable decode-runtime diagnostics artifacts."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DECODE_PROFILE_PROMPTS = (
    "Introduce yourself in one sentence.",
    "List three properties of prime numbers.",
)
_TOP_LEVEL_TIMING_FIELDS = (
    "scheduler_s",
    "prepare_decode_s",
    "model_execute_s",
    "sampler_s",
    "postprocess_s",
)
_MODEL_EXECUTE_SUBCOMPONENT_FIELDS = (
    "kv_update_s",
    "partitioned_decode_reduction_s",
)
_MANIFEST_ENV_PREFIXES = (
    "NANOVLLM_JAX_",
    "JAX_",
    "XLA_",
)


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _serialize_output_records(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        token_ids = [int(token) for token in output.get("token_ids", [])]
        records.append(
            {
                "index": int(index),
                "token_ids": token_ids,
                "text": str(output.get("text", "")),
            }
        )
    return records


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        value_key = str(value)
        counts[value_key] = counts.get(value_key, 0) + 1
    return counts


def _timing_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            float_value = float(value)
            if math.isfinite(float_value):
                values.append(float_value)
    return values


def _summarize_metric_values(
    values: list[float],
    *,
    top_level_total_s: float | None = None,
    model_execute_total_s: float | None = None,
) -> dict[str, Any]:
    total_s = float(sum(values))
    records_with_value = len(values)
    mean_s = (total_s / records_with_value) if records_with_value else None
    max_s = max(values) if values else None
    share_of_top_level_pct = None
    if top_level_total_s is not None and top_level_total_s > 0.0:
        share_of_top_level_pct = (total_s / top_level_total_s) * 100.0
    share_of_model_execute_pct = None
    if model_execute_total_s is not None and model_execute_total_s > 0.0:
        share_of_model_execute_pct = (total_s / model_execute_total_s) * 100.0
    return {
        "records_with_value": records_with_value,
        "total_s": total_s,
        "mean_s": mean_s,
        "max_s": max_s,
        "share_of_top_level_pct": share_of_top_level_pct,
        "share_of_model_execute_pct": share_of_model_execute_pct,
    }


def _summarize_decode_step_timings(
    decode_step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    top_level_values = {
        field: _timing_values(decode_step_records, field)
        for field in _TOP_LEVEL_TIMING_FIELDS
    }
    top_level_total_s = sum(sum(values) for values in top_level_values.values())
    model_execute_total_s = sum(top_level_values["model_execute_s"])
    return {
        "decode_step_count": len(decode_step_records),
        "top_level_total_s": float(top_level_total_s),
        "model_execute_total_s": float(model_execute_total_s),
        "top_level": {
            field: _summarize_metric_values(
                values,
                top_level_total_s=top_level_total_s,
            )
            for field, values in top_level_values.items()
        },
        "model_execute_subcomponents": {
            field: _summarize_metric_values(
                _timing_values(decode_step_records, field),
                top_level_total_s=top_level_total_s,
                model_execute_total_s=model_execute_total_s,
            )
            for field in _MODEL_EXECUTE_SUBCOMPONENT_FIELDS
        },
    }


def _summarize_kv_update_counters(
    decode_step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    total_calls = int(
        sum(int(record.get("kv_update_calls", 0) or 0) for record in decode_step_records)
    )
    total_tokens = int(
        sum(int(record.get("kv_update_tokens", 0) or 0) for record in decode_step_records)
    )
    total_valid_tokens = int(
        sum(
            int(record.get("kv_update_valid_tokens", 0) or 0)
            for record in decode_step_records
        )
    )
    total_skipped_tokens = int(
        sum(
            int(record.get("kv_update_skipped_tokens", 0) or 0)
            for record in decode_step_records
        )
    )
    total_duplicate_slots = int(
        sum(
            int(record.get("kv_update_duplicate_slots", 0) or 0)
            for record in decode_step_records
        )
    )
    valid_token_pct = None
    skipped_token_pct = None
    duplicate_slot_pct = None
    if total_tokens > 0:
        valid_token_pct = (total_valid_tokens / total_tokens) * 100.0
        skipped_token_pct = (total_skipped_tokens / total_tokens) * 100.0
    if total_valid_tokens > 0:
        duplicate_slot_pct = (total_duplicate_slots / total_valid_tokens) * 100.0
    return {
        "kv_update_calls": total_calls,
        "kv_update_tokens": total_tokens,
        "kv_update_valid_tokens": total_valid_tokens,
        "kv_update_skipped_tokens": total_skipped_tokens,
        "kv_update_duplicate_slots": total_duplicate_slots,
        "kv_update_valid_token_pct": valid_token_pct,
        "kv_update_skipped_token_pct": skipped_token_pct,
        "kv_update_duplicate_slot_pct": duplicate_slot_pct,
    }


def _aggregate_kv_update_counters(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    total_calls = int(
        sum(
            int(summary.get("counters", {}).get("kv_update_calls", 0) or 0)
            for summary in summaries
        )
    )
    total_tokens = int(
        sum(
            int(summary.get("counters", {}).get("kv_update_tokens", 0) or 0)
            for summary in summaries
        )
    )
    total_valid_tokens = int(
        sum(
            int(summary.get("counters", {}).get("kv_update_valid_tokens", 0) or 0)
            for summary in summaries
        )
    )
    total_skipped_tokens = int(
        sum(
            int(summary.get("counters", {}).get("kv_update_skipped_tokens", 0) or 0)
            for summary in summaries
        )
    )
    total_duplicate_slots = int(
        sum(
            int(summary.get("counters", {}).get("kv_update_duplicate_slots", 0) or 0)
            for summary in summaries
        )
    )
    valid_token_pct = None
    skipped_token_pct = None
    duplicate_slot_pct = None
    if total_tokens > 0:
        valid_token_pct = (total_valid_tokens / total_tokens) * 100.0
        skipped_token_pct = (total_skipped_tokens / total_tokens) * 100.0
    if total_valid_tokens > 0:
        duplicate_slot_pct = (total_duplicate_slots / total_valid_tokens) * 100.0
    return {
        "kv_update_calls": total_calls,
        "kv_update_tokens": total_tokens,
        "kv_update_valid_tokens": total_valid_tokens,
        "kv_update_skipped_tokens": total_skipped_tokens,
        "kv_update_duplicate_slots": total_duplicate_slots,
        "kv_update_valid_token_pct": valid_token_pct,
        "kv_update_skipped_token_pct": skipped_token_pct,
        "kv_update_duplicate_slot_pct": duplicate_slot_pct,
    }


def _compare_kv_update_counters(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    before = before or {}
    after = after or {}
    fields = (
        "kv_update_calls",
        "kv_update_tokens",
        "kv_update_valid_tokens",
        "kv_update_skipped_tokens",
        "kv_update_duplicate_slots",
        "kv_update_valid_token_pct",
        "kv_update_skipped_token_pct",
        "kv_update_duplicate_slot_pct",
    )
    return {
        field: _diff_numeric_metric(before.get(field), after.get(field))
        for field in fields
    }


def _diff_numeric_metric(before: Any, after: Any) -> dict[str, Any]:
    before_value = None if before is None else float(before)
    after_value = None if after is None else float(after)
    delta = None
    if before_value is not None and after_value is not None:
        delta = after_value - before_value
    return {
        "before": before_value,
        "after": after_value,
        "delta": delta,
    }


def _compare_metric_summary(
    before_metric: dict[str, Any] | None,
    after_metric: dict[str, Any] | None,
) -> dict[str, Any]:
    before_metric = before_metric or {}
    after_metric = after_metric or {}
    return {
        "records_with_value": _diff_numeric_metric(
            before_metric.get("records_with_value"),
            after_metric.get("records_with_value"),
        ),
        "total_s": _diff_numeric_metric(
            before_metric.get("total_s"),
            after_metric.get("total_s"),
        ),
        "mean_s": _diff_numeric_metric(
            before_metric.get("mean_s"),
            after_metric.get("mean_s"),
        ),
        "max_s": _diff_numeric_metric(
            before_metric.get("max_s"),
            after_metric.get("max_s"),
        ),
        "share_of_top_level_pct": _diff_numeric_metric(
            before_metric.get("share_of_top_level_pct"),
            after_metric.get("share_of_top_level_pct"),
        ),
        "share_of_model_execute_pct": _diff_numeric_metric(
            before_metric.get("share_of_model_execute_pct"),
            after_metric.get("share_of_model_execute_pct"),
        ),
    }


def _aggregate_metric_summaries(
    metrics: list[dict[str, Any] | None],
    *,
    top_level_total_s: float | None = None,
    model_execute_total_s: float | None = None,
) -> dict[str, Any]:
    present = [metric or {} for metric in metrics]
    total_s = float(sum(float(metric.get("total_s", 0.0) or 0.0) for metric in present))
    records_with_value = int(
        sum(int(metric.get("records_with_value", 0) or 0) for metric in present)
    )
    max_values = [
        float(metric["max_s"])
        for metric in present
        if metric.get("max_s") is not None
    ]
    mean_s = (total_s / records_with_value) if records_with_value else None
    max_s = max(max_values) if max_values else None
    share_of_top_level_pct = None
    if top_level_total_s is not None and top_level_total_s > 0.0:
        share_of_top_level_pct = (total_s / top_level_total_s) * 100.0
    share_of_model_execute_pct = None
    if model_execute_total_s is not None and model_execute_total_s > 0.0:
        share_of_model_execute_pct = (total_s / model_execute_total_s) * 100.0
    return {
        "records_with_value": records_with_value,
        "total_s": total_s,
        "mean_s": mean_s,
        "max_s": max_s,
        "share_of_top_level_pct": share_of_top_level_pct,
        "share_of_model_execute_pct": share_of_model_execute_pct,
    }


def summarize_decode_profile_runs(
    summary_paths: list[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Aggregate timing-oriented profile summaries across multiple runs."""
    if not summary_paths:
        raise ValueError("summarize_decode_profile_runs requires at least one summary path")

    summaries = [_load_summary(path) for path in summary_paths]
    aggregate_top_level_total_s = float(
        sum(
            float(summary.get("timings", {}).get("top_level_total_s", 0.0) or 0.0)
            for summary in summaries
        )
    )
    aggregate_model_execute_total_s = float(
        sum(
            float(summary.get("timings", {}).get("model_execute_total_s", 0.0) or 0.0)
            for summary in summaries
        )
    )
    return {
        "format_version": 1,
        "run_count": len(summaries),
        "summary_paths": [summary["summary_path"] for summary in summaries],
        "timings": {
            "decode_step_count": int(
                sum(
                    int(summary.get("timings", {}).get("decode_step_count", 0) or 0)
                    for summary in summaries
                )
            ),
            "top_level_total_s": aggregate_top_level_total_s,
            "model_execute_total_s": aggregate_model_execute_total_s,
            "top_level": {
                field: _aggregate_metric_summaries(
                    [summary.get("timings", {}).get("top_level", {}).get(field) for summary in summaries],
                    top_level_total_s=aggregate_top_level_total_s,
                )
                for field in _TOP_LEVEL_TIMING_FIELDS
            },
            "model_execute_subcomponents": {
                field: _aggregate_metric_summaries(
                    [
                        summary.get("timings", {}).get("model_execute_subcomponents", {}).get(field)
                        for summary in summaries
                    ],
                    top_level_total_s=aggregate_top_level_total_s,
                    model_execute_total_s=aggregate_model_execute_total_s,
                )
                for field in _MODEL_EXECUTE_SUBCOMPONENT_FIELDS
            },
        },
        "counters": _aggregate_kv_update_counters(summaries),
        "runs": [
            {
                "summary_path": summary["summary_path"],
                "decode_attention_backend": summary.get("runtime", {}).get("decode_attention_backend"),
                "kv_update_backend": summary.get("runtime", {}).get("kv_update_backend"),
                "decode_step_count": int(summary.get("timings", {}).get("decode_step_count", 0) or 0),
                "top_level_total_s": float(summary.get("timings", {}).get("top_level_total_s", 0.0) or 0.0),
            }
            for summary in summaries
        ],
    }


def _load_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    summary_path = Path(path).expanduser()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Decode profile summary must be a JSON object: {summary_path}")
    payload["summary_path"] = str(summary_path)
    return payload


def _run_command(args: list[str], *, cwd: str | os.PathLike[str] | None = None) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = result.stdout.strip()
    return output or None


def _collect_git_manifest() -> dict[str, Any]:
    repo_root = _run_command(["git", "rev-parse", "--show-toplevel"])
    sha = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run_command(["git", "status", "--porcelain"], cwd=repo_root)
    return {
        "repo_root": repo_root,
        "sha": sha,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _collect_env_manifest() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in sorted(os.environ):
        if key.startswith(_MANIFEST_ENV_PREFIXES):
            env[key] = os.environ[key]
    return env


def _build_run_manifest(
    *,
    runtime: dict[str, Any],
    invocation: dict[str, Any] | None,
    env_manifest: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _collect_git_manifest(),
        "env": dict(env_manifest or {}),
        "runtime": runtime,
        "invocation": dict(invocation or {}),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
    }


def _diff_counts(
    before_counts: dict[str, Any] | None,
    after_counts: dict[str, Any] | None,
) -> dict[str, dict[str, int]]:
    before_counts = before_counts or {}
    after_counts = after_counts or {}
    diff: dict[str, dict[str, int]] = {}
    keys = sorted({str(key) for key in before_counts} | {str(key) for key in after_counts})
    for key in keys:
        before_value = int(before_counts.get(key, 0))
        after_value = int(after_counts.get(key, 0))
        diff[key] = {
            "before": before_value,
            "after": after_value,
            "delta": after_value - before_value,
        }
    return diff


def compare_decode_profile_summaries(
    before_path: str | os.PathLike[str],
    after_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load two decode profile summaries and return a stable JSON-friendly diff."""
    before = _load_summary(before_path)
    after = _load_summary(after_path)

    before_artifacts = before.get("artifacts", {})
    after_artifacts = after.get("artifacts", {})
    before_histograms = before.get("histograms", {})
    after_histograms = after.get("histograms", {})
    before_timings = before.get("timings", {})
    after_timings = after.get("timings", {})

    return {
        "format_version": 1,
        "before_summary_path": before["summary_path"],
        "after_summary_path": after["summary_path"],
        "record_counts": {
            "decode_step_records": {
                "before": int(before_artifacts.get("decode_step_records", 0)),
                "after": int(after_artifacts.get("decode_step_records", 0)),
                "delta": int(after_artifacts.get("decode_step_records", 0))
                - int(before_artifacts.get("decode_step_records", 0)),
            },
            "decode_schedule_records": {
                "before": int(before_artifacts.get("decode_schedule_records", 0)),
                "after": int(after_artifacts.get("decode_schedule_records", 0)),
                "delta": int(after_artifacts.get("decode_schedule_records", 0))
                - int(before_artifacts.get("decode_schedule_records", 0)),
            },
        },
        "histograms": {
            "decode_input_actions": _diff_counts(
                before_histograms.get("decode_input_actions"),
                after_histograms.get("decode_input_actions"),
            ),
            "prepared_metadata_actions": _diff_counts(
                before_histograms.get("prepared_metadata_actions"),
                after_histograms.get("prepared_metadata_actions"),
            ),
            "block_table_actions": _diff_counts(
                before_histograms.get("block_table_actions"),
                after_histograms.get("block_table_actions"),
            ),
            "kv_update_backends": _diff_counts(
                before_histograms.get("kv_update_backends"),
                after_histograms.get("kv_update_backends"),
            ),
            "kv_update_measured": _diff_counts(
                before_histograms.get("kv_update_measured"),
                after_histograms.get("kv_update_measured"),
            ),
            "partitioned_decode_reduction_backends": _diff_counts(
                before_histograms.get("partitioned_decode_reduction_backends"),
                after_histograms.get("partitioned_decode_reduction_backends"),
            ),
            "partitioned_decode_reduction_families": _diff_counts(
                before_histograms.get("partitioned_decode_reduction_families"),
                after_histograms.get("partitioned_decode_reduction_families"),
            ),
            "partitioned_decode_reduction_measured": _diff_counts(
                before_histograms.get("partitioned_decode_reduction_measured"),
                after_histograms.get("partitioned_decode_reduction_measured"),
            ),
            "decode_schedule_actions": _diff_counts(
                before_histograms.get("decode_schedule_actions"),
                after_histograms.get("decode_schedule_actions"),
            ),
        },
        "context": {
            "before_runtime": before.get("runtime", {}),
            "after_runtime": after.get("runtime", {}),
            "before_prompt_count": int(before.get("prompts", {}).get("count", 0)),
            "after_prompt_count": int(after.get("prompts", {}).get("count", 0)),
            "before_output_count": int(before.get("outputs", {}).get("count", 0)),
            "after_output_count": int(after.get("outputs", {}).get("count", 0)),
        },
        "counters": _compare_kv_update_counters(
            before.get("counters"),
            after.get("counters"),
        ),
        "timings": {
            "decode_step_count": _diff_numeric_metric(
                before_timings.get("decode_step_count"),
                after_timings.get("decode_step_count"),
            ),
            "top_level_total_s": _diff_numeric_metric(
                before_timings.get("top_level_total_s"),
                after_timings.get("top_level_total_s"),
            ),
            "model_execute_total_s": _diff_numeric_metric(
                before_timings.get("model_execute_total_s"),
                after_timings.get("model_execute_total_s"),
            ),
            "top_level": {
                field: _compare_metric_summary(
                    before_timings.get("top_level", {}).get(field),
                    after_timings.get("top_level", {}).get(field),
                )
                for field in _TOP_LEVEL_TIMING_FIELDS
            },
            "model_execute_subcomponents": {
                field: _compare_metric_summary(
                    before_timings.get("model_execute_subcomponents", {}).get(field),
                    after_timings.get("model_execute_subcomponents", {}).get(field),
                )
                for field in _MODEL_EXECUTE_SUBCOMPONENT_FIELDS
            },
        },
    }


def run_controlled_decode_profile(
    *,
    model_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    prompts: list[str] | None = None,
    max_tokens: int = 32,
    temperature: float = 0.0,
    decode_attention_backend: str = "auto",
    kv_update_backend: str | None = None,
    mosaic_kernel_family: str | None = None,
    mosaic_min_decode_batch: int | None = None,
    mosaic_throughput_min_decode_batch: int | None = None,
    enforce_eager: bool = False,
    extra_env_overrides: dict[str, str] | None = None,
    invocation: dict[str, Any] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    llm_class=None,
    sampling_params_cls=None,
) -> dict[str, Any]:
    """Run a controlled decode workload and write stable diagnostics artifacts."""
    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    output_dir = Path(output_dir).expanduser()
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    decode_step_path = raw_dir / "decode_step_profile.jsonl"
    decode_schedule_path = raw_dir / "decode_schedule.jsonl"
    for path in (decode_step_path, decode_schedule_path):
        if path.exists():
            path.unlink()

    resolved_prompts = list(prompts or DEFAULT_DECODE_PROFILE_PROMPTS)
    if not resolved_prompts:
        raise ValueError("run_controlled_decode_profile requires at least one prompt")

    llm_init_kwargs = dict(llm_kwargs or {})
    llm_init_kwargs.setdefault("tensor_parallel_size", 1)
    llm_init_kwargs.setdefault("max_num_seqs", max(1, len(resolved_prompts)))
    llm_init_kwargs.setdefault("decode_attention_backend", decode_attention_backend)
    llm_init_kwargs.setdefault("enforce_eager", enforce_eager)

    env_overrides = {
        "NANOVLLM_JAX_PROFILE_DECODE_STEP": "1",
        "NANOVLLM_JAX_DUMP_DECODE_SCHEDULE": "1",
        "NANOVLLM_JAX_DIAGNOSTICS_DIR": str(raw_dir),
    }
    if kv_update_backend is not None:
        env_overrides["NANOVLLM_JAX_KV_UPDATE_BACKEND"] = str(kv_update_backend)
    if mosaic_kernel_family is not None:
        env_overrides["NANOVLLM_JAX_MOSAIC_DECODE_KERNEL"] = str(
            mosaic_kernel_family
        )
    if mosaic_min_decode_batch is not None:
        env_overrides["NANOVLLM_JAX_INTERNAL_MOSAIC_MIN_DECODE_BATCH"] = str(
            int(mosaic_min_decode_batch)
        )
    if mosaic_throughput_min_decode_batch is not None:
        env_overrides["NANOVLLM_JAX_INTERNAL_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH"] = str(
            int(mosaic_throughput_min_decode_batch)
        )
    if extra_env_overrides is not None:
        env_overrides.update(
            {str(key): str(value) for key, value in extra_env_overrides.items()}
        )

    outputs = []
    llm = None
    env_manifest: dict[str, str] = {}
    with _temporary_env(env_overrides):
        env_manifest = _collect_env_manifest()
        resolved_llm_class = llm_class
        resolved_sampling_params_cls = sampling_params_cls
        if resolved_llm_class is None or resolved_sampling_params_cls is None:
            from nanovllm_jax import LLM, SamplingParams

            resolved_llm_class = resolved_llm_class or LLM
            resolved_sampling_params_cls = resolved_sampling_params_cls or SamplingParams
        llm = resolved_llm_class(str(model_path), **llm_init_kwargs)
        sampling_params = resolved_sampling_params_cls(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = llm.generate(
            resolved_prompts,
            sampling_params,
            use_tqdm=False,
        )

    if llm is not None and hasattr(llm, "exit"):
        llm.exit()

    decode_step_records = _read_jsonl(decode_step_path)
    decode_schedule_records = _read_jsonl(decode_schedule_path)
    runtime = {
        "decode_attention_backend": decode_attention_backend,
        "kv_update_backend": (
            str(kv_update_backend) if kv_update_backend is not None else "default"
        ),
        "mosaic_kernel_family": (
            str(mosaic_kernel_family) if mosaic_kernel_family is not None else "default"
        ),
        "mosaic_min_decode_batch": (
            int(mosaic_min_decode_batch)
            if mosaic_min_decode_batch is not None
            else "default"
        ),
        "mosaic_throughput_min_decode_batch": (
            int(mosaic_throughput_min_decode_batch)
            if mosaic_throughput_min_decode_batch is not None
            else "default"
        ),
        "enforce_eager": bool(llm_init_kwargs["enforce_eager"]),
        "tensor_parallel_size": int(llm_init_kwargs["tensor_parallel_size"]),
        "max_num_seqs": int(llm_init_kwargs["max_num_seqs"]),
    }
    manifest = _build_run_manifest(
        runtime=runtime,
        invocation=invocation,
        env_manifest=env_manifest,
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "raw_dir": str(raw_dir),
        "runtime": runtime,
        "sampling": {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        "prompts": {
            "count": len(resolved_prompts),
            "items": resolved_prompts,
        },
        "outputs": {
            "count": len(outputs),
            "token_counts": [len(output.get("token_ids", [])) for output in outputs],
            "records": _serialize_output_records(outputs),
        },
        "artifacts": {
            "decode_step_profile_path": str(decode_step_path),
            "decode_schedule_path": str(decode_schedule_path),
            "run_manifest_path": str(manifest_path),
            "decode_step_records": len(decode_step_records),
            "decode_schedule_records": len(decode_schedule_records),
        },
        "histograms": {
            "decode_schedule_actions": _count_by(decode_schedule_records, "action"),
            "decode_input_actions": _count_by(decode_step_records, "decode_input_action"),
            "prepared_metadata_actions": _count_by(
                decode_step_records,
                "prepared_metadata_action",
            ),
            "block_table_actions": _count_by(decode_step_records, "block_table_action"),
            "kv_update_backends": _count_by(decode_step_records, "kv_update_backend"),
            "kv_update_measured": _count_by(decode_step_records, "kv_update_measured"),
            "partitioned_decode_reduction_backends": _count_by(
                decode_step_records,
                "partitioned_decode_reduction_backend",
            ),
            "partitioned_decode_reduction_families": _count_by(
                decode_step_records,
                "partitioned_decode_reduction_family",
            ),
            "partitioned_decode_reduction_measured": _count_by(
                decode_step_records,
                "partitioned_decode_reduction_measured",
            ),
        },
        "counters": _summarize_kv_update_counters(decode_step_records),
        "timings": _summarize_decode_step_timings(decode_step_records),
        "manifest": manifest,
    }

    summary_path = output_dir / "decode_runtime_profile_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def run_kv_update_backend_matrix(
    *,
    model_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    backends: list[str] | None = None,
    baseline_backend: str = "scatter",
    prompts: list[str] | None = None,
    max_tokens: int = 32,
    temperature: float = 0.0,
    decode_attention_backend: str = "auto",
    mosaic_kernel_family: str | None = None,
    mosaic_min_decode_batch: int | None = None,
    mosaic_throughput_min_decode_batch: int | None = None,
    enforce_eager: bool = False,
    invocation: dict[str, Any] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    llm_class=None,
    sampling_params_cls=None,
) -> dict[str, Any]:
    """Run the controlled decode profile across multiple KV-update backends."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if backends is None:
        resolved_backends = [
            "scatter",
            "compact_scatter",
            "sorted_compact_scatter",
        ]
    else:
        resolved_backends = list(backends)
    if not resolved_backends:
        raise ValueError("run_kv_update_backend_matrix requires at least one backend")
    if baseline_backend not in resolved_backends:
        raise ValueError(
            f"baseline backend {baseline_backend!r} must be included in backends: {resolved_backends}"
        )

    runs: list[dict[str, Any]] = []
    for backend in resolved_backends:
        backend_output_dir = output_dir / backend
        backend_invocation = dict(invocation or {})
        backend_invocation["kv_update_backend"] = backend
        runs.append(
            run_controlled_decode_profile(
                model_path=model_path,
                output_dir=backend_output_dir,
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                decode_attention_backend=decode_attention_backend,
                kv_update_backend=backend,
                mosaic_kernel_family=mosaic_kernel_family,
                mosaic_min_decode_batch=mosaic_min_decode_batch,
                mosaic_throughput_min_decode_batch=mosaic_throughput_min_decode_batch,
                enforce_eager=enforce_eager,
                invocation=backend_invocation,
                llm_kwargs=llm_kwargs,
                llm_class=llm_class,
                sampling_params_cls=sampling_params_cls,
            )
        )

    run_by_backend = {
        str(run.get("runtime", {}).get("kv_update_backend")): run
        for run in runs
    }
    baseline_run = run_by_backend[baseline_backend]
    baseline_summary_path = baseline_run["summary_path"]

    comparisons_vs_baseline: dict[str, dict[str, Any]] = {}
    for backend in resolved_backends:
        if backend == baseline_backend:
            continue
        candidate_run = run_by_backend[backend]
        comparison = compare_decode_profile_summaries(
            baseline_summary_path,
            candidate_run["summary_path"],
        )
        comparison_path = output_dir / f"compare_{baseline_backend}_vs_{backend}.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        comparisons_vs_baseline[backend] = {
            "comparison_path": str(comparison_path),
            "kv_update_total_s_delta": comparison["timings"]["model_execute_subcomponents"][
                "kv_update_s"
            ]["total_s"]["delta"],
            "kv_update_valid_tokens_delta": comparison["counters"][
                "kv_update_valid_tokens"
            ]["delta"],
            "kv_update_duplicate_slots_delta": comparison["counters"][
                "kv_update_duplicate_slots"
            ]["delta"],
            "kv_update_share_of_model_execute_pct_delta": comparison["timings"][
                "model_execute_subcomponents"
            ]["kv_update_s"]["share_of_model_execute_pct"]["delta"],
        }

    aggregate = summarize_decode_profile_runs(
        [run["summary_path"] for run in runs]
    )

    matrix = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(model_path).expanduser()),
        "output_dir": str(output_dir),
        "baseline_backend": baseline_backend,
        "backend_order": resolved_backends,
        "runtime": {
            "decode_attention_backend": decode_attention_backend,
            "mosaic_kernel_family": (
                str(mosaic_kernel_family) if mosaic_kernel_family is not None else "default"
            ),
            "mosaic_min_decode_batch": (
                int(mosaic_min_decode_batch)
                if mosaic_min_decode_batch is not None
                else "default"
            ),
            "mosaic_throughput_min_decode_batch": (
                int(mosaic_throughput_min_decode_batch)
                if mosaic_throughput_min_decode_batch is not None
                else "default"
            ),
            "enforce_eager": bool(enforce_eager),
        },
        "runs": [
            {
                "backend": str(run.get("runtime", {}).get("kv_update_backend")),
                "summary_path": run["summary_path"],
                "output_dir": str(Path(run["output_dir"])),
                "decode_step_count": int(run.get("timings", {}).get("decode_step_count", 0) or 0),
                "kv_update_total_s": run.get("timings", {})
                .get("model_execute_subcomponents", {})
                .get("kv_update_s", {})
                .get("total_s"),
                "kv_update_share_of_model_execute_pct": run.get("timings", {})
                .get("model_execute_subcomponents", {})
                .get("kv_update_s", {})
                .get("share_of_model_execute_pct"),
                "counters": dict(run.get("counters", {})),
            }
            for run in runs
        ],
        "aggregate": aggregate,
        "comparisons_vs_baseline": comparisons_vs_baseline,
    }

    matrix_path = output_dir / "kv_update_backend_matrix.json"
    matrix_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    matrix["matrix_path"] = str(matrix_path)
    return matrix
