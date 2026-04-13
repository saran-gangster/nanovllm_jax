#!/usr/bin/env python3
"""Run the real-model throughput-v2 runtime gate with deterministic prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nanovllm_jax.utils.decode_profile_artifacts import run_controlled_decode_profile
from nanovllm_jax.utils.throughput_v2_gate import (
    DEFAULT_RUNTIME_GATE_PROMPTS,
    summarize_runtime_gate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile blockwise vs throughput_v2_mosaic on deterministic prompts.",
    )
    repo_root = Path(__file__).resolve().parent
    default_model = repo_root / "models/qwen/Qwen-3-0.6B"
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL_PATH", str(default_model)),
        help="Local HuggingFace model directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Deterministic prompt to include. Repeat for multiple prompts.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0001,
        help="Must stay above the runtime minimum.",
    )
    parser.add_argument("--mosaic-min-decode-batch", type=int, default=0)
    parser.add_argument("--mosaic-throughput-min-decode-batch", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = list(args.prompts or DEFAULT_RUNTIME_GATE_PROMPTS)

    blockwise_summary = run_controlled_decode_profile(
        model_path=args.model,
        output_dir=output_dir / "blockwise",
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        decode_attention_backend="blockwise",
        enforce_eager=args.enforce_eager,
        invocation={
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "entrypoint": "profile_throughput_v2_runtime_gate.py",
            "variant": "blockwise",
        },
    )
    throughput_v2_summary = run_controlled_decode_profile(
        model_path=args.model,
        output_dir=output_dir / "throughput_v2_mosaic",
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        decode_attention_backend="mosaic",
        mosaic_kernel_family="throughput_v2",
        mosaic_min_decode_batch=args.mosaic_min_decode_batch,
        mosaic_throughput_min_decode_batch=args.mosaic_throughput_min_decode_batch,
        enforce_eager=args.enforce_eager,
        extra_env_overrides={
            "NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "1",
            "NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
        },
        invocation={
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "entrypoint": "profile_throughput_v2_runtime_gate.py",
            "variant": "throughput_v2_mosaic",
        },
    )

    runtime_gate = summarize_runtime_gate(
        blockwise_summary=blockwise_summary,
        throughput_v2_summary=throughput_v2_summary,
    )
    runtime_gate["prompts"] = prompts
    runtime_gate["gate"] = {
        "requires_token_identity": True,
        "requires_top_level_improvement": (
            runtime_gate["timings"]["top_level_total_s"]["delta"] is not None
            and runtime_gate["timings"]["top_level_total_s"]["delta"] < 0.0
        ),
        "requires_model_execute_improvement": (
            runtime_gate["timings"]["model_execute_total_s"]["delta"] is not None
            and runtime_gate["timings"]["model_execute_total_s"]["delta"] < 0.0
        ),
    }

    summary_path = output_dir / "runtime_gate_summary.json"
    _write_json(summary_path, runtime_gate)
    print(summary_path)


if __name__ == "__main__":
    main()
