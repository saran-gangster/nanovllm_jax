#!/usr/bin/env python3
"""Run the throughput-v2 split-policy speed window in one GPU session."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanovllm_jax.utils.decode_kernel_bench import (
    build_kernel_benchmark_manifest,
    now_utc_iso,
)
from nanovllm_jax.utils.throughput_v2_gate import (
    DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES,
    PromotionGateRow,
    build_speed_window_split_rows,
    build_splitk_override_table,
    merge_splitk_override_tables,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run quick split screens, confirmation split sweeps, the 10-row "
            "promotion gate, and optional real-model runtime gate for "
            "throughput_v2_mosaic."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-rows",
        default="speed_window",
        help=(
            "Rows to sweep. Use 'speed_window' or comma-separated "
            "batchxhead_dimxblocks entries, for example 512x128x64,1024x128x32."
        ),
    )
    parser.add_argument(
        "--split-candidates",
        default=",".join(str(value) for value in DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES),
    )
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick-warmup", type=int, default=5)
    parser.add_argument("--quick-iters", type=int, default=20)
    parser.add_argument("--quick-repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--matrix", choices=("primary", "extended"), default="extended")
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL_PATH", "/workspace/models/Qwen3-0.6B"),
        help="Local HuggingFace model path for the runtime gate.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--skip-runtime-gate", action="store_true")
    parser.add_argument("--include-jax-reference", action="store_true")
    parser.add_argument(
        "--verify-against-blockwise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_split_candidates(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("split candidates must be >= 1")
        values.append(value)
    if not values:
        raise ValueError("--split-candidates must include at least one integer")
    return tuple(dict.fromkeys(values))


def _parse_split_rows(
    raw: str,
    *,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: str,
) -> list[PromotionGateRow]:
    if str(raw).strip().lower() in {"speed_window", "default"}:
        base_rows = build_speed_window_split_rows()
        return [
            PromotionGateRow(
                row.batch_size,
                row.head_dim,
                row.max_blocks_per_seq,
                block_size=block_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dtype=dtype,
            )
            for row in base_rows
        ]

    rows: list[PromotionGateRow] = []
    for token in str(raw).split(","):
        token = token.strip().lower().replace("b", "", 1)
        if not token:
            continue
        parts = token.replace(":", "x").replace("/", "x").split("x")
        if len(parts) not in {3, 4}:
            raise ValueError(
                "--split-rows entries must look like batchxhead_dimxblocks"
            )
        row_block_size = int(parts[3]) if len(parts) == 4 else block_size
        rows.append(
            PromotionGateRow(
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
                block_size=row_block_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dtype=dtype,
            )
        )
    if not rows:
        raise ValueError("--split-rows did not resolve to any rows")
    return rows


def _python_script_command(repo_root: Path, script_name: str) -> list[str]:
    return [sys.executable, str(repo_root / script_name)]


def _run_command(
    command: list[str],
    *,
    repo_root: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if extra_env:
        env.update(extra_env)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _split_sweep_command(
    *,
    repo_root: Path,
    output_dir: Path,
    row: PromotionGateRow,
    split_candidates: tuple[int, ...],
    warmup: int,
    iters: int,
    repetitions: int,
    num_blocks: int,
    seed: int,
    include_jax_reference: bool,
    verify_against_blockwise: bool,
) -> list[str]:
    command = _python_script_command(repo_root, "bench_throughput_v2_split_sweep.py")
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(row.batch_size),
            "--head-dim",
            str(row.head_dim),
            "--max-blocks-per-seq",
            str(row.max_blocks_per_seq),
            "--block-size",
            str(row.block_size),
            "--num-heads",
            str(row.num_heads),
            "--num-kv-heads",
            str(row.num_kv_heads),
            "--num-blocks",
            str(num_blocks),
            "--dtype",
            row.dtype,
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--repetitions",
            str(repetitions),
            "--split-candidates",
            ",".join(str(value) for value in split_candidates),
            "--include-current-mosaic",
        ]
    )
    if include_jax_reference:
        command.append("--include-jax-reference")
    if verify_against_blockwise:
        command.append("--verify-against-blockwise")
    else:
        command.append("--no-verify-against-blockwise")
    return command


def _promotion_gate_command(
    *,
    repo_root: Path,
    output_dir: Path,
    matrix: str,
    num_heads: int,
    num_kv_heads: int,
    num_blocks: int,
    dtype: str,
    seed: int,
    warmup: int,
    iters: int,
    repetitions: int,
    include_jax_reference: bool,
    verify_against_blockwise: bool,
) -> list[str]:
    command = _python_script_command(repo_root, "bench_throughput_v2_promotion_gate.py")
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--matrix",
            matrix,
            "--num-heads",
            str(num_heads),
            "--num-kv-heads",
            str(num_kv_heads),
            "--num-blocks",
            str(num_blocks),
            "--dtype",
            dtype,
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--repetitions",
            str(repetitions),
        ]
    )
    if include_jax_reference:
        command.append("--include-jax-reference")
    if verify_against_blockwise:
        command.append("--verify-against-blockwise")
    else:
        command.append("--no-verify-against-blockwise")
    return command


def _runtime_gate_command(
    *,
    repo_root: Path,
    output_dir: Path,
    model: str,
    max_tokens: int,
) -> list[str]:
    command = _python_script_command(repo_root, "profile_throughput_v2_runtime_gate.py")
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--model",
            model,
            "--max-tokens",
            str(max_tokens),
        ]
    )
    return command


def _winner_split(summary: dict[str, Any]) -> int | None:
    winner = summary.get("winner", {}) or {}
    if not winner.get("qualified"):
        return None
    split_k = winner.get("split_k")
    return int(split_k) if split_k is not None else None


def _winner_improvement(summary: dict[str, Any]) -> dict[str, Any] | None:
    winner = summary.get("winner", {}) or {}
    if not winner.get("qualified"):
        return None
    mean_ms = winner.get("mean_of_means_ms")
    if mean_ms is None:
        return None
    gate = summary.get("gate", {}) or {}
    mean_ms = float(mean_ms)
    blockwise_ms = gate.get("blockwise_mean_of_means_ms")
    current_ms = gate.get("current_mosaic_default_mean_of_means_ms")
    return {
        "split_k": winner.get("split_k"),
        "mean_of_means_ms": mean_ms,
        "blockwise_mean_of_means_ms": blockwise_ms,
        "current_mosaic_default_mean_of_means_ms": current_ms,
        "speedup_vs_blockwise_pct": (
            ((float(blockwise_ms) - mean_ms) / float(blockwise_ms)) * 100.0
            if blockwise_ms is not None and float(blockwise_ms) != 0.0
            else None
        ),
        "speedup_vs_current_mosaic_default_pct": (
            ((float(current_ms) - mean_ms) / float(current_ms)) * 100.0
            if current_ms is not None and float(current_ms) != 0.0
            else None
        ),
    }


def _gate_passed(summary: dict[str, Any], *, gate_name: str) -> bool:
    gate = summary.get("gate", {}) or {}
    if gate_name == "promotion":
        return bool(gate.get("all_rows_passed"))
    if gate_name == "runtime":
        return bool(gate.get("passes_runtime_gate"))
    raise ValueError(f"unknown gate: {gate_name}")


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_candidates = _parse_split_candidates(args.split_candidates)
    split_rows = _parse_split_rows(
        args.split_rows,
        block_size=args.block_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        dtype=args.dtype,
    )

    manifest = build_kernel_benchmark_manifest(
        invocation={
            "argv": sys.argv,
            "entrypoint": "bench_throughput_v2_speed_window.py",
            "args": vars(args),
            "split_rows": [row.shape_key for row in split_rows],
            "split_candidates": split_candidates,
        },
        runtime={
            "repo_root": str(repo_root),
            "cwd": str(Path.cwd()),
        },
    )
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)

    quick_summaries: dict[str, str] = {}
    confirmation_summaries: dict[str, str] = {}
    confirmed_override_tables: list[dict[str, int]] = []
    confirmed_winners: dict[str, dict[str, Any]] = {}
    confirmed_rankings: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for row in split_rows:
        quick_dir = output_dir / "split_sweep_quick" / row.shape_key
        quick_command = _split_sweep_command(
            repo_root=repo_root,
            output_dir=quick_dir,
            row=row,
            split_candidates=split_candidates,
            warmup=args.quick_warmup,
            iters=args.quick_iters,
            repetitions=args.quick_repetitions,
            num_blocks=args.num_blocks,
            seed=args.seed,
            include_jax_reference=args.include_jax_reference,
            verify_against_blockwise=args.verify_against_blockwise,
        )
        commands.append({"phase": "quick_split_sweep", "row": row.shape_key, "command": quick_command})
        _run_command(quick_command, repo_root=repo_root)
        quick_summary_path = quick_dir / "summary.json"
        quick_summaries[row.shape_key] = str(quick_summary_path)
        quick_summary = _read_json(quick_summary_path)
        quick_winner = _winner_split(quick_summary)
        if quick_winner is None:
            continue

        confirmation_dir = output_dir / "split_sweep_confirmed" / row.shape_key
        confirmation_command = _split_sweep_command(
            repo_root=repo_root,
            output_dir=confirmation_dir,
            row=row,
            split_candidates=(quick_winner,),
            warmup=args.warmup,
            iters=args.iters,
            repetitions=args.repetitions,
            num_blocks=args.num_blocks,
            seed=args.seed,
            include_jax_reference=args.include_jax_reference,
            verify_against_blockwise=args.verify_against_blockwise,
        )
        commands.append(
            {
                "phase": "confirmed_split_sweep",
                "row": row.shape_key,
                "command": confirmation_command,
            }
        )
        _run_command(confirmation_command, repo_root=repo_root)
        confirmation_summary_path = confirmation_dir / "summary.json"
        confirmation_summaries[row.shape_key] = str(confirmation_summary_path)
        confirmation_summary = _read_json(confirmation_summary_path)
        confirmed_winner = _winner_split(confirmation_summary)
        if confirmed_winner is None:
            continue
        confirmed_override_tables.append(
            build_splitk_override_table(split_k=confirmed_winner, row=row)
        )
        confirmed_winners[row.shape_key] = {
            "split_k": confirmed_winner,
            "summary_path": str(confirmation_summary_path),
            "winner": confirmation_summary.get("winner"),
        }
        improvement = _winner_improvement(confirmation_summary)
        if improvement is not None:
            confirmed_rankings.append({"shape_key": row.shape_key, **improvement})

    confirmed_rankings.sort(
        key=lambda item: (
            item.get("speedup_vs_current_mosaic_default_pct")
            if item.get("speedup_vs_current_mosaic_default_pct") is not None
            else -1.0
        ),
        reverse=True,
    )

    merged_splitk_table = merge_splitk_override_tables(confirmed_override_tables)
    splitk_table_path = output_dir / "throughput_v2_splitk_override_candidate.json"
    _write_json(splitk_table_path, merged_splitk_table)

    promotion_dir = output_dir / "promotion_gate"
    promotion_command = _promotion_gate_command(
        repo_root=repo_root,
        output_dir=promotion_dir,
        matrix=args.matrix,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_blocks=args.num_blocks,
        dtype=args.dtype,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        repetitions=args.repetitions,
        include_jax_reference=args.include_jax_reference,
        verify_against_blockwise=args.verify_against_blockwise,
    )
    commands.append({"phase": "promotion_gate", "command": promotion_command})
    _run_command(
        promotion_command,
        repo_root=repo_root,
        extra_env={
            "NANOVLLM_JAX_MOSAIC_THROUGHPUT_SPLITK_TABLE_PATH": str(splitk_table_path),
        },
    )
    promotion_summary_path = promotion_dir / "summary.json"
    promotion_summary = _read_json(promotion_summary_path)

    runtime_summary_path: Path | None = None
    runtime_summary: dict[str, Any] | None = None
    if not args.skip_runtime_gate:
        runtime_dir = output_dir / "runtime_gate"
        runtime_command = _runtime_gate_command(
            repo_root=repo_root,
            output_dir=runtime_dir,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        commands.append({"phase": "runtime_gate", "command": runtime_command})
        _run_command(
            runtime_command,
            repo_root=repo_root,
            extra_env={
                "NANOVLLM_JAX_MOSAIC_THROUGHPUT_SPLITK_TABLE_PATH": str(splitk_table_path),
            },
        )
        runtime_summary_path = runtime_dir / "runtime_gate_summary.json"
        runtime_summary = _read_json(runtime_summary_path)

    final_gate = {
        "promotion_gate_passed": _gate_passed(promotion_summary, gate_name="promotion"),
        "runtime_gate_required": not args.skip_runtime_gate,
        "runtime_gate_passed": (
            True
            if args.skip_runtime_gate
            else _gate_passed(runtime_summary or {}, gate_name="runtime")
        ),
        "has_confirmed_split_overrides": bool(merged_splitk_table),
        "portability_validation_skipped": True,
    }
    final_gate["ready_to_apply_candidate_split_table"] = (
        final_gate["promotion_gate_passed"]
        and final_gate["runtime_gate_passed"]
        and final_gate["has_confirmed_split_overrides"]
    )

    summary = {
        "format_version": 1,
        "generated_at_utc": now_utc_iso(),
        "manifest_path": str(manifest_path),
        "quick_split_summaries": quick_summaries,
        "confirmation_split_summaries": confirmation_summaries,
        "confirmed_winners": confirmed_winners,
        "confirmed_rankings": confirmed_rankings,
        "candidate_splitk_override_path": str(splitk_table_path),
        "candidate_splitk_override_table": merged_splitk_table,
        "promotion_summary_path": str(promotion_summary_path),
        "runtime_summary_path": str(runtime_summary_path) if runtime_summary_path else None,
        "commands": commands,
        "gate": final_gate,
    }
    summary_path = output_dir / "speed_window_summary.json"
    _write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
