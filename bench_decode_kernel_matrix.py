#!/usr/bin/env python3
"""Run strict fresh-process A/B benchmarks for synthetic decode kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanovllm_jax.utils.decode_kernel_bench import (
    build_kernel_benchmark_manifest,
    build_worker_command,
    compare_kernel_benchmark_summaries,
    extract_case_env,
    now_utc_iso,
    parse_case_spec,
    run_worker_command,
    summarize_kernel_case_runs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict fresh-process A/B benchmarking for synthetic decode kernels.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case spec like 'name=baseline,family=baseline,block_q=64,block_kv=64'. Repeat per case.",
    )
    parser.add_argument(
        "--baseline-case",
        required=True,
        help="Case name used as the comparison baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-blocks-per-seq", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick-warmup", type=int, default=5)
    parser.add_argument("--quick-iters", type=int, default=20)
    parser.add_argument("--stability-warmup", type=int, default=20)
    parser.add_argument("--stability-iters", type=int, default=200)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--verify-against-blockwise", action="store_true")
    return parser.parse_args()


def _common_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "max_blocks_per_seq": args.max_blocks_per_seq,
        "num_blocks": args.num_blocks,
        "dtype": args.dtype,
        "seed": args.seed,
    }


def _run_pass(
    *,
    repo_root: Path,
    output_dir: Path,
    pass_name: str,
    warmup: int,
    iters: int,
    cases: list[dict[str, Any]],
    common_args: dict[str, Any],
    baseline_case: str,
    repetitions: int,
    verify_against_blockwise: bool,
) -> dict[str, Any]:
    pass_dir = output_dir / pass_name
    pass_dir.mkdir(parents=True, exist_ok=True)
    case_summary_paths: dict[str, str] = {}

    for case in cases:
        case_name = str(case["name"])
        case_dir = pass_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        run_paths: list[str] = []
        commands: list[list[str]] = []
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
            run_worker_command(
                command,
                repo_root=repo_root,
                extra_env=extract_case_env(case),
            )
            run_paths.append(str(run_path))

        summary = summarize_kernel_case_runs(
            run_paths,
            case_name=case_name,
            pass_name=pass_name,
        )
        summary["commands"] = commands
        summary_path = case_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        case_summary_paths[case_name] = str(summary_path)

    baseline_summary_path = case_summary_paths[baseline_case]
    comparison_paths: dict[str, str] = {}
    for case_name, summary_path in case_summary_paths.items():
        if case_name == baseline_case:
            continue
        comparison = compare_kernel_benchmark_summaries(baseline_summary_path, summary_path)
        comparison_path = pass_dir / f"compare_{baseline_case}_vs_{case_name}.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        comparison_paths[case_name] = str(comparison_path)

    pass_summary = {
        "format_version": 1,
        "generated_at_utc": now_utc_iso(),
        "pass_name": pass_name,
        "warmup": warmup,
        "iters": iters,
        "repetitions": repetitions,
        "baseline_case": baseline_case,
        "case_summaries": case_summary_paths,
        "comparisons_vs_baseline": comparison_paths,
    }
    pass_summary_path = pass_dir / "pass_summary.json"
    pass_summary_path.write_text(
        json.dumps(pass_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pass_summary["path"] = str(pass_summary_path)
    return pass_summary


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [parse_case_spec(spec) for spec in args.case]
    case_names = [str(case["name"]) for case in cases]
    if len(set(case_names)) != len(case_names):
        raise ValueError("Case names must be unique")
    if args.baseline_case not in case_names:
        raise ValueError("--baseline-case must match one of the provided case names")

    manifest = build_kernel_benchmark_manifest(
        invocation={
            "args": vars(args),
            "cases": cases,
        },
        runtime={
            "repo_root": str(repo_root),
            "cwd": str(Path.cwd()),
        },
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pass_summaries: list[dict[str, Any]] = []
    pass_summaries.append(
        _run_pass(
            repo_root=repo_root,
            output_dir=output_dir,
            pass_name="quick",
            warmup=args.quick_warmup,
            iters=args.quick_iters,
            cases=cases,
            common_args=_common_args(args),
            baseline_case=args.baseline_case,
            repetitions=args.repetitions,
            verify_against_blockwise=args.verify_against_blockwise,
        )
    )
    if not args.skip_stability:
        pass_summaries.append(
            _run_pass(
                repo_root=repo_root,
                output_dir=output_dir,
                pass_name="stability",
                warmup=args.stability_warmup,
                iters=args.stability_iters,
                cases=cases,
                common_args=_common_args(args),
                baseline_case=args.baseline_case,
                repetitions=args.repetitions,
                verify_against_blockwise=args.verify_against_blockwise,
            )
        )

    matrix = {
        "format_version": 1,
        "generated_at_utc": now_utc_iso(),
        "manifest_path": str(manifest_path),
        "baseline_case": args.baseline_case,
        "cases": case_names,
        "passes": {summary["pass_name"]: summary for summary in pass_summaries},
    }
    matrix_path = output_dir / "decode_kernel_bench_matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(matrix_path)


if __name__ == "__main__":
    main()
