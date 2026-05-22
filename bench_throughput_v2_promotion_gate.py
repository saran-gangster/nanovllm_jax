#!/usr/bin/env python3
"""Run the synthetic throughput-v2 promotion gate in fresh processes."""

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
    PromotionGateRow,
    build_canary_kernel_table,
    build_promotion_gate_cases,
    build_promotion_gate_rows,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the throughput-v2 synthetic promotion gate row by row.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--matrix",
        choices=("primary", "extended"),
        default="extended",
        help="Promotion-gate row set to execute.",
    )
    parser.add_argument("--include-jax-reference", action="store_true")
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
    parser.add_argument("--verify-against-blockwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--write-canary-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a candidate rollout table for rows that pass the gate.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _row_with_layout(row: PromotionGateRow, args: argparse.Namespace) -> PromotionGateRow:
    return PromotionGateRow(
        batch_size=row.batch_size,
        head_dim=row.head_dim,
        max_blocks_per_seq=row.max_blocks_per_seq,
        block_size=row.block_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        dtype=args.dtype,
    )


def _run_case(
    *,
    repo_root: Path,
    row_dir: Path,
    pass_name: str,
    row: PromotionGateRow,
    case: dict[str, Any],
    common_args: dict[str, Any],
    warmup: int,
    iters: int,
    repetitions: int,
    verify_against_blockwise: bool,
) -> tuple[dict[str, Any], Path]:
    case_name = str(case["name"])
    case_dir = row_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    run_paths: list[str] = []
    commands: list[list[str]] = []
    failures: list[dict[str, Any]] = []

    for rep in range(1, repetitions + 1):
        run_path = case_dir / f"run_{rep:03d}.json"
        command = build_worker_command(
            repo_root=repo_root,
            common_args=common_args,
            case=case,
            output_json=run_path,
            warmup=warmup,
            iters=iters,
            verify_against_blockwise=verify_against_blockwise,
        )
        commands.append(command)
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
            "pass_name": pass_name,
            "family": case.get("family"),
            "shape": {
                "batch_size": row.batch_size,
                "head_dim": row.head_dim,
                "max_blocks_per_seq": row.max_blocks_per_seq,
                "block_size": row.block_size,
            },
            "failed": True,
            "run_paths": run_paths,
            "commands": commands,
            "failures": failures,
        }
    else:
        summary = summarize_kernel_case_runs(
            run_paths,
            case_name=case_name,
            pass_name=pass_name,
        )
        summary["commands"] = commands
        summary["failed"] = False

    summary_path = case_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary, summary_path


def _run_row(
    *,
    repo_root: Path,
    output_dir: Path,
    row: PromotionGateRow,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_dir = output_dir / row.shape_key
    row_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: dict[str, dict[str, Any]] = {}
    case_summary_paths: dict[str, str] = {}
    comparisons: dict[str, str] = {}
    baseline_summary_path: Path | None = None

    for case in cases:
        summary, summary_path = _run_case(
            repo_root=repo_root,
            row_dir=row_dir,
            pass_name=args.matrix,
            row=row,
            case=case,
            common_args=_common_args(args, row),
            warmup=args.warmup,
            iters=args.iters,
            repetitions=args.repetitions,
            verify_against_blockwise=args.verify_against_blockwise,
        )
        case_name = str(case["name"])
        case_summaries[case_name] = summary
        case_summary_paths[case_name] = str(summary_path)
        if case_name == "blockwise":
            if summary.get("failed"):
                raise RuntimeError(
                    f"Baseline case failed for {row.shape_key}; see {summary_path}"
                )
            baseline_summary_path = summary_path

    if baseline_summary_path is None:
        raise ValueError("Promotion gate requires a blockwise baseline case")

    for case_name, summary_path in case_summary_paths.items():
        if case_name == "blockwise":
            continue
        if case_summaries[case_name].get("failed"):
            continue
        comparison = compare_kernel_benchmark_summaries(
            baseline_summary_path,
            summary_path,
        )
        comparison_path = row_dir / f"compare_blockwise_vs_{case_name}.json"
        _write_json(comparison_path, comparison)
        comparisons[case_name] = str(comparison_path)

    blockwise_summary = case_summaries["blockwise"]
    throughput_v2_summary = case_summaries.get("throughput_v2_mosaic")
    throughput_v2_mean = None
    blockwise_mean = float(blockwise_summary["timings"]["mean_of_means_ms"])
    throughput_v2_diff = None
    throughput_v2_pass = None
    throughput_v2_verify = None
    throughput_v2_all_finite = None
    if throughput_v2_summary is not None and not throughput_v2_summary.get("failed"):
        throughput_v2_mean = float(throughput_v2_summary["timings"]["mean_of_means_ms"])
        throughput_v2_diff = (
            throughput_v2_summary.get("verify", {}) or {}
        ).get("max_abs_diff")
        throughput_v2_verify = throughput_v2_diff
        throughput_v2_all_finite = bool(
            (throughput_v2_summary.get("output", {}) or {}).get("all_finite", False)
        )
        throughput_v2_pass = (
            throughput_v2_mean < blockwise_mean
            and throughput_v2_diff is not None
            and float(throughput_v2_diff) <= 1e-3
            and throughput_v2_all_finite
        )

    row_summary = {
        "format_version": 1,
        "shape_key": row.shape_key,
        "row": {
            "batch_size": row.batch_size,
            "head_dim": row.head_dim,
            "max_blocks_per_seq": row.max_blocks_per_seq,
            "block_size": row.block_size,
        },
        "case_summaries": case_summary_paths,
        "comparisons_vs_blockwise": comparisons,
        "gate": {
            "blockwise_mean_of_means_ms": blockwise_mean,
            "throughput_v2_mosaic_mean_of_means_ms": throughput_v2_mean,
            "throughput_v2_mosaic_max_abs_diff": throughput_v2_verify,
            "throughput_v2_mosaic_all_finite": throughput_v2_all_finite,
            "throughput_v2_mosaic_passes_row_gate": throughput_v2_pass,
        },
    }
    _write_json(row_dir / "row_summary.json", row_summary)
    return row_summary


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_row_with_layout(row, args) for row in build_promotion_gate_rows(args.matrix)]
    cases = build_promotion_gate_cases(include_jax_reference=args.include_jax_reference)
    manifest = build_kernel_benchmark_manifest(
        invocation={
            "argv": sys.argv,
            "entrypoint": "bench_throughput_v2_promotion_gate.py",
            "args": vars(args),
            "rows": [row.shape_key for row in rows],
            "cases": cases,
        },
        runtime={
            "repo_root": str(repo_root),
            "cwd": str(Path.cwd()),
        },
    )
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)

    row_summaries: list[dict[str, Any]] = []
    promoted_rows: list[PromotionGateRow] = []
    for row in rows:
        row_summary = _run_row(
            repo_root=repo_root,
            output_dir=output_dir,
            row=row,
            cases=cases,
            args=args,
        )
        row_summaries.append(row_summary)
        if row_summary["gate"]["throughput_v2_mosaic_passes_row_gate"]:
            promoted_rows.append(row)

    canary_table = build_canary_kernel_table(promoted_rows)
    canary_table_path = output_dir / "throughput_v2_canary_table_candidate.json"
    if args.write_canary_table:
        _write_json(canary_table_path, canary_table)

    summary = {
        "format_version": 1,
        "generated_at_utc": now_utc_iso(),
        "manifest_path": str(manifest_path),
        "matrix": args.matrix,
        "rows": {row_summary["shape_key"]: row_summary for row_summary in row_summaries},
        "candidate_rollout_rows": [row.shape_key for row in promoted_rows],
        "gate": {
            "row_count": len(rows),
            "passed_row_count": len(promoted_rows),
            "all_rows_passed": len(promoted_rows) == len(rows),
            "requires_candidate_faster_than_blockwise": True,
            "requires_max_abs_diff_lte": 1e-3,
            "requires_all_outputs_finite": True,
        },
        "candidate_rollout_table_path": (
            str(canary_table_path) if args.write_canary_table else None
        ),
        "candidate_rollout_table": canary_table if args.write_canary_table else None,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
