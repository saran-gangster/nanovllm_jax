"""Run a controlled decode profile across KV-update backends."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nanovllm_jax.utils.decode_profile_artifacts import (
    run_kv_update_backend_matrix,
)


def _parse_backend_list(raw: str) -> list[str]:
    backends = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not backends:
        raise argparse.ArgumentTypeError("backend list must not be empty")
    return backends


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled decode profiles across KV-update backends.",
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
        default=str(repo_root / "kv_update_backend_out"),
        help="Directory for per-backend artifacts and matrix JSON.",
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
        "--backends",
        type=_parse_backend_list,
        default=_parse_backend_list("scatter,compact_scatter,sorted_compact_scatter"),
        help="Comma-separated KV-update backends to profile.",
    )
    parser.add_argument(
        "--baseline-backend",
        choices=("scatter", "compact_scatter", "sorted_compact_scatter"),
        default="scatter",
        help="Baseline backend used for generated comparisons.",
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
    matrix = run_kv_update_backend_matrix(
        model_path=args.model,
        output_dir=args.output_dir,
        backends=args.backends,
        baseline_backend=args.baseline_backend,
        prompts=args.prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        decode_attention_backend=args.decode_backend,
        mosaic_kernel_family=args.mosaic_kernel_family,
        mosaic_min_decode_batch=args.mosaic_min_decode_batch,
        mosaic_throughput_min_decode_batch=args.mosaic_throughput_min_decode_batch,
        enforce_eager=args.enforce_eager,
        invocation={
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "entrypoint": "profile_kv_update_backends.py",
        },
    )
    print(json.dumps(matrix, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
