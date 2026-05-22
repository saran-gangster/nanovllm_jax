"""Helpers for strict synthetic decode-kernel A/B benchmarking."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanovllm_jax.utils.decode_profile_artifacts import (
    _build_run_manifest,
    _collect_env_manifest,
)


_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}
_BOOLEAN_OPTIONAL_ARGS = {"use_schedule_barrier"}
_ENV_CASE_PREFIX = "env__"


def _coerce_case_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    try:
        if raw.startswith("0") and raw not in {"0", "0.0"} and not raw.startswith("0."):
            raise ValueError
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_case_spec(spec: str) -> dict[str, Any]:
    """Parse a case string like ``name=baseline,family=baseline,block_q=64``."""
    parsed: dict[str, Any] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid case token: {token!r}")
        key, value = token.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Invalid case token: {token!r}")
        parsed[key] = _coerce_case_value(value.strip())
    if "name" not in parsed or "family" not in parsed:
        raise ValueError("Each --case must include name=... and family=...")
    return parsed


def extract_case_env(case: dict[str, Any]) -> dict[str, str]:
    """Extract environment overrides from ``env__FOO=...`` case entries."""
    env: dict[str, str] = {}
    for key, value in case.items():
        if not key.startswith(_ENV_CASE_PREFIX):
            continue
        env_key = key[len(_ENV_CASE_PREFIX) :]
        if not env_key:
            continue
        if isinstance(value, bool):
            env[env_key] = "1" if value else "0"
        else:
            env[env_key] = str(value)
    return env


def build_worker_command(
    *,
    repo_root: str | os.PathLike[str],
    common_args: dict[str, Any],
    case: dict[str, Any],
    output_json: str | os.PathLike[str],
    warmup: int,
    iters: int,
    verify_against_blockwise: bool = False,
) -> list[str]:
    """Build the fresh-process bench worker command for one case."""
    repo_root = str(repo_root)
    script_path = str(Path(repo_root) / "bench_decode_families.py")
    command = [sys.executable, script_path]
    merged = dict(common_args)
    merged.update(
        {
            key: value
            for key, value in case.items()
            if key not in {"name"} and not key.startswith(_ENV_CASE_PREFIX)
        }
    )
    merged["output_json"] = str(output_json)
    merged["warmup"] = warmup
    merged["iters"] = iters
    if verify_against_blockwise:
        merged["verify_against_blockwise"] = True

    for key, value in merged.items():
        cli_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(cli_key)
            elif key in _BOOLEAN_OPTIONAL_ARGS:
                command.append(f"--no-{key.replace('_', '-')}")
            continue
        command.extend([cli_key, str(value)])
    return command


def load_kernel_benchmark(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_kernel_case_runs(
    paths: list[str | os.PathLike[str]],
    *,
    case_name: str,
    pass_name: str,
) -> dict[str, Any]:
    """Summarize one case across one benchmark pass."""
    if not paths:
        raise ValueError("paths must not be empty")
    records = [load_kernel_benchmark(path) for path in paths]
    means = [float(record["timings"]["mean_ms"]) for record in records]
    p50s = [float(record["timings"]["p50_ms"]) for record in records]
    mins = [float(record["timings"]["min_ms"]) for record in records]
    maxes = [float(record["timings"]["max_ms"]) for record in records]
    compiles = [float(record["timings"]["compile_and_first_run_s"]) for record in records]
    checksum = records[0].get("output", {}).get("checksum_f32_sum")
    output_records = [record.get("output", {}) for record in records]
    nonfinite_count = sum(
        int(record.get("nonfinite_count", 0) or 0)
        for record in output_records
    )
    verify = records[0].get("verify")
    return {
        "format_version": 1,
        "case_name": case_name,
        "pass_name": pass_name,
        "family": records[0].get("family"),
        "shape": records[0].get("shape", {}),
        "dtype": records[0].get("dtype"),
        "run_count": len(records),
        "run_paths": [str(Path(path)) for path in paths],
        "timings": {
            "mean_of_means_ms": sum(means) / len(means),
            "mean_of_p50_ms": sum(p50s) / len(p50s),
            "best_min_ms": min(mins),
            "worst_max_ms": max(maxes),
            "mean_compile_s": sum(compiles) / len(compiles),
        },
        "output": {
            "checksum_f32_sum": checksum,
            "all_finite": all(
                bool(record.get("all_finite", True))
                for record in output_records
            ),
            "nonfinite_count": nonfinite_count,
        },
        "verify": verify,
        "family_notes": records[0].get("family_notes", {}),
    }


def compare_kernel_benchmark_summaries(
    before_path: str | os.PathLike[str],
    after_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Compare two summarized kernel benchmark JSON files."""
    before = load_kernel_benchmark(before_path)
    after = load_kernel_benchmark(after_path)
    before_timings = before.get("timings", {})
    after_timings = after.get("timings", {})
    fields = (
        "mean_of_means_ms",
        "mean_of_p50_ms",
        "best_min_ms",
        "worst_max_ms",
        "mean_compile_s",
    )
    comparison: dict[str, Any] = {
        "format_version": 1,
        "before_summary_path": str(Path(before_path)),
        "after_summary_path": str(Path(after_path)),
        "before_case_name": before.get("case_name"),
        "after_case_name": after.get("case_name"),
        "pass_name": before.get("pass_name"),
        "timings": {},
    }
    for field in fields:
        before_value = float(before_timings.get(field, 0.0))
        after_value = float(after_timings.get(field, 0.0))
        comparison["timings"][field] = {
            "before": before_value,
            "after": after_value,
            "delta": after_value - before_value,
            "ratio_vs_before": (
                after_value / before_value if before_value not in (0.0, -0.0) else None
            ),
        }
    return comparison


def build_kernel_benchmark_manifest(
    *,
    invocation: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime or {})
    if "cwd" not in runtime:
        runtime["cwd"] = os.getcwd()
    return _build_run_manifest(
        runtime=runtime,
        invocation=invocation,
        env_manifest=_collect_env_manifest(),
    )


def run_worker_command(
    command: list[str],
    *,
    repo_root: str | os.PathLike[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one synthetic kernel benchmark worker in a fresh subprocess."""
    repo_root = str(repo_root)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(repo_root) / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
