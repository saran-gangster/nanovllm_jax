#!/usr/bin/env python3
"""Sweep throughput-v2 split policy for one throughput-v2 row."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanovllm_jax.utils.decode_kernel_bench import (
    build_kernel_benchmark_manifest,
    build_worker_command,
    compare_kernel_benchmark_summaries,
    extract_case_env,
    now_utc_iso,
    run_worker_command,
    summarize_kernel_case_runs,
)
from nanovllm_jax.utils.throughput_v2_gate import (
    DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES,
    DEFAULT_SPLIT_SWEEP_ROW,
    PromotionGateRow,
    build_split_sweep_cases,
    build_splitk_override_table,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep throughput-v2 split-k choices for a fixed shape.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SPLIT_SWEEP_ROW.batch_size)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_SPLIT_SWEEP_ROW.head_dim)
    parser.add_argument(
        "--max-blocks-per-seq",
        type=int,
        default=DEFAULT_SPLIT_SWEEP_ROW.max_blocks_per_seq,
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_SPLIT_SWEEP_ROW.block_size)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--split-candidates",
        default=",".join(str(value) for value in DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES),
    )
    parser.add_argument(
        "--include-current-mosaic",
        action="store_true",
        help="Compare forced split candidates against the current default throughput_v2_mosaic policy.",
    )
    parser.add_argument("--include-jax-reference", action="store_true")
    parser.add_argument("--verify-against-blockwise", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_split_candidates(raw: str) -> tuple[int, ...]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--split-candidates must include at least one integer")
    return tuple(values)


def _case_mean_ms(summary: dict[str, Any] | None) -> float | None:
    if summary is None or summary.get("failed"):
        return None
    timings = summary.get("timings", {}) or {}
    value = timings.get("mean_of_means_ms")
    return float(value) if value is not None else None


def _forced_split_from_case_name(case_name: str) -> int | None:
    prefix = "throughput_v2_mosaic_split"
    if not case_name.startswith(prefix):
        return None
    return int(case_name.removeprefix(prefix))


def _common_args(args: argparse.Namespace, row: PromotionGateRow) -> dict[str, Any]:
    return {
        "batch_size": row.batch_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": row.head_dim,
        "block_size": row.block_size,
        "max_blocks_per_seq": row.max_blocks_per_seq,
        "num_blocks": args.num_blocks,
        "dtype": args.dtype,
        "seed": args.seed,
    }


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    row = PromotionGateRow(
        batch_size=args.batch_size,
        head_dim=args.head_dim,
        max_blocks_per_seq=args.max_blocks_per_seq,
        block_size=args.block_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        dtype=args.dtype,
    )
    split_candidates = _parse_split_candidates(args.split_candidates)
    cases = build_split_sweep_cases(
        split_candidates=split_candidates,
        include_current_mosaic=args.include_current_mosaic,
        include_jax_reference=args.include_jax_reference,
    )

    manifest = build_kernel_benchmark_manifest(
        invocation={
            "argv": sys.argv,
            "entrypoint": "bench_throughput_v2_split_sweep.py",
            "args": vars(args),
            "cases": cases,
            "shape_key": row.shape_key,
        },
        runtime={
            "repo_root": str(repo_root),
            "cwd": str(Path.cwd()),
        },
    )
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)

    case_summaries: dict[str, dict[str, Any]] = {}
    case_summary_paths: dict[str, str] = {}
    commands: dict[str, list[list[str]]] = {}

    for case in cases:
        case_name = str(case["name"])
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        run_paths: list[str] = []
        commands[case_name] = []
        failures: list[dict[str, Any]] = []
        for rep in range(1, args.repetitions + 1):
            run_path = case_dir / f"run_{rep:03d}.json"
            command = build_worker_command(
                repo_root=repo_root,
                common_args=_common_args(args, row),
                case=case,
                output_json=run_path,
                warmup=args.warmup,
                iters=args.iters,
                verify_against_blockwise=args.verify_against_blockwise,
            )
            commands[case_name].append(command)
            try:
                run_worker_command(
                    command,
                    repo_root=repo_root,
                    extra_env=extract_case_env(case),
                )
            except subprocess.CalledProcessError as exc:
                failures.append(
                    {
                        "run_index": rep,
                        "command": command,
                        "returncode": int(exc.returncode),
                        "stdout": exc.stdout,
                        "stderr": exc.stderr,
                    }
                )
                break
            run_paths.append(str(run_path))

        if failures:
            summary = {
                "format_version": 1,
                "case_name": case_name,
                "pass_name": "split_sweep",
                "family": case.get("family"),
                "shape": {
                    "batch_size": row.batch_size,
                    "head_dim": row.head_dim,
                    "max_blocks_per_seq": row.max_blocks_per_seq,
                    "block_size": row.block_size,
                },
                "failed": True,
                "run_paths": run_paths,
                "commands": commands[case_name],
                "failures": failures,
            }
        else:
            summary = summarize_kernel_case_runs(
                run_paths,
                case_name=case_name,
                pass_name="split_sweep",
            )
            summary["commands"] = commands[case_name]
            summary["failed"] = False

        summary_path = case_dir / "summary.json"
        _write_json(summary_path, summary)
        case_summaries[case_name] = summary
        case_summary_paths[case_name] = str(summary_path)

    baseline_summary = case_summaries["blockwise"]
    if baseline_summary.get("failed"):
        raise RuntimeError(f"Baseline case failed; see {case_summary_paths['blockwise']}")

    comparisons: dict[str, str] = {}
    for case_name, summary_path in case_summary_paths.items():
        if case_name == "blockwise" or case_summaries[case_name].get("failed"):
            continue
        comparison = compare_kernel_benchmark_summaries(
            case_summary_paths["blockwise"],
            summary_path,
        )
        comparison_path = output_dir / f"compare_blockwise_vs_{case_name}.json"
        _write_json(comparison_path, comparison)
        comparisons[case_name] = str(comparison_path)

    winner_name: str | None = None
    winner_split_k: int | None = None
    winner_mean_ms: float | None = None
    winner_diff = None
    winner_all_finite = None
    blockwise_mean_ms = float(baseline_summary["timings"]["mean_of_means_ms"])
    current_mosaic_summary = case_summaries.get("throughput_v2_mosaic_default")
    current_mosaic_mean_ms = _case_mean_ms(current_mosaic_summary)
    comparison_mean_ms = [
        value
        for value in (blockwise_mean_ms, current_mosaic_mean_ms)
        if value is not None
    ]
    for case_name, summary in case_summaries.items():
        forced_split = _forced_split_from_case_name(case_name)
        if forced_split is None:
            continue
        if summary.get("failed"):
            continue
        mean_ms = float(summary["timings"]["mean_of_means_ms"])
        max_abs_diff = (summary.get("verify", {}) or {}).get("max_abs_diff")
        all_finite = bool((summary.get("output", {}) or {}).get("all_finite", False))
        if (
            any(mean_ms >= comparison_mean for comparison_mean in comparison_mean_ms)
            or max_abs_diff is None
            or float(max_abs_diff) > 1e-3
            or not all_finite
        ):
            continue
        if winner_mean_ms is None or mean_ms < winner_mean_ms:
            winner_name = case_name
            winner_mean_ms = mean_ms
            winner_split_k = forced_split
            winner_diff = max_abs_diff
            winner_all_finite = all_finite

    override_table = None
    override_table_path = None
    if winner_split_k is not None:
        override_table = build_splitk_override_table(split_k=winner_split_k, row=row)
        override_table_path = output_dir / "throughput_v2_splitk_override_candidate.json"
        _write_json(override_table_path, override_table)

    summary = {
        "format_version": 1,
        "generated_at_utc": now_utc_iso(),
        "manifest_path": str(manifest_path),
        "shape_key": row.shape_key,
        "case_summaries": case_summary_paths,
        "comparisons_vs_blockwise": comparisons,
        "winner": {
            "case_name": winner_name,
            "split_k": winner_split_k,
            "mean_of_means_ms": winner_mean_ms,
            "max_abs_diff": winner_diff,
            "all_finite": winner_all_finite,
            "qualified": winner_split_k is not None,
        },
        "gate": {
            "blockwise_mean_of_means_ms": blockwise_mean_ms,
            "current_mosaic_default_mean_of_means_ms": current_mosaic_mean_ms,
            "requires_candidate_faster_than_blockwise": True,
            "requires_candidate_faster_than_current_mosaic_default": (
                current_mosaic_mean_ms is not None
            ),
            "requires_max_abs_diff_lte": 1e-3,
            "requires_all_outputs_finite": True,
        },
        "candidate_splitk_override_path": (
            str(override_table_path) if override_table_path is not None else None
        ),
        "candidate_splitk_override_table": override_table,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
