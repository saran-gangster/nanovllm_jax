"""Run a controlled decode workload and emit stable runtime diagnostics artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nanovllm_jax.utils.decode_profile_artifacts import run_controlled_decode_profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled decode workload and write runtime diagnostics artifacts.",
    )
    repo_root = Path(__file__).resolve().parent
    default_model = repo_root / "models/qwen/Qwen-3-0.6B"
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL_PATH", str(default_model)),
        help="Local HuggingFace model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "decode_profile_out"),
        help="Directory for raw diagnostics and summary JSON.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to include. Repeat for multiple prompts. Defaults are used when omitted.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Generation length per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=("auto", "blockwise", "mosaic"),
        default="auto",
        help="Public decode backend selection.",
    )
    parser.add_argument(
        "--kv-update-backend",
        choices=("scatter", "compact_scatter", "sorted_compact_scatter"),
        help="Internal KV-update backend override for controlled A/B profiling.",
    )
    parser.add_argument(
        "--mosaic-kernel-family",
        choices=("auto", "baseline", "latency", "throughput"),
        help="Internal Mosaic family override for controlled profiling.",
    )
    parser.add_argument(
        "--mosaic-min-decode-batch",
        type=int,
        help="Internal Mosaic min padded batch override for controlled profiling.",
    )
    parser.add_argument(
        "--mosaic-throughput-min-decode-batch",
        type=int,
        help="Internal throughput-family min padded batch override for controlled profiling.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable JIT compilation for the profiling run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_controlled_decode_profile(
        model_path=args.model,
        output_dir=args.output_dir,
        prompts=args.prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        decode_attention_backend=args.decode_backend,
        kv_update_backend=args.kv_update_backend,
        mosaic_kernel_family=args.mosaic_kernel_family,
        mosaic_min_decode_batch=args.mosaic_min_decode_batch,
        mosaic_throughput_min_decode_batch=args.mosaic_throughput_min_decode_batch,
        enforce_eager=args.enforce_eager,
        invocation={
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "entrypoint": "profile_decode_runtime.py",
        },
    )
    print(f"Summary: {summary['summary_path']}")
    print(f"Manifest: {summary['artifacts']['run_manifest_path']}")
    print(f"Decode step records: {summary['artifacts']['decode_step_records']}")
    print(f"Decode schedule records: {summary['artifacts']['decode_schedule_records']}")


if __name__ == "__main__":
    main()
