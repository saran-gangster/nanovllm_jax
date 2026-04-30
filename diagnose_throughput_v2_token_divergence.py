#!/usr/bin/env python3
"""Diagnose first-token divergence between blockwise and throughput-v2 decode.

The public entry point runs each candidate in a fresh Python process so that
import-time Mosaic flags are not shared across variants.  If a token mismatch
is found, it reruns each variant to the first divergent generation step, saves
the logits for that step, and writes a compact comparison packet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROMPTS = (
    "Summarize the word latency in one short sentence.",
    "Summarize the phrase throughput optimization in one short sentence.",
    "Summarize the phrase decode kernel in one short sentence.",
)
VARIANT_ORDER = ("blockwise", "throughput_v2_jax", "throughput_v2_mosaic")


@dataclass(frozen=True)
class VariantConfig:
    name: str
    decode_attention_backend: str
    env: dict[str, str]


VARIANTS: dict[str, VariantConfig] = {
    "blockwise": VariantConfig(
        name="blockwise",
        decode_attention_backend="blockwise",
        env={},
    ),
    "throughput_v2_jax": VariantConfig(
        name="throughput_v2_jax",
        decode_attention_backend="mosaic",
        env={
            "NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
            "NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "0",
        },
    ),
    "throughput_v2_mosaic": VariantConfig(
        name="throughput_v2_mosaic",
        decode_attention_backend="mosaic",
        env={
            "NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
            "NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "1",
        },
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare blockwise, throughput-v2 JAX, and throughput-v2 Mosaic logits.",
    )
    repo_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "NANOVLLM_MODEL_PATH",
            str(repo_root / "models/qwen/Qwen-3-0.6B"),
        ),
        help="Local HuggingFace model directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--workload",
        choices=("tiny", "canary", "both"),
        default="both",
        help="Run the tiny failed batch, the canary-shaped batch, or both.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Base prompt to include. Repeated for canary workloads.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0001,
        help="Match the runtime gate by default. Use <1e-6 for greedy diagnostics.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--canary-batch-size", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--mosaic-min-decode-batch", type=int, default=0)
    parser.add_argument("--mosaic-throughput-min-decode-batch", type=int, default=0)
    parser.add_argument("--splitk-table-path", default="")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--keep-full-logits",
        action="store_true",
        help="Keep .npy full-logit files after computing max diffs.",
    )

    # Worker mode is intentionally hidden from normal help; it is used by this
    # script to isolate variant-specific env/import state.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANT_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--prompts-json", help=argparse.SUPPRESS)
    parser.add_argument("--save-logits-step", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--stop-after-generation-step",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-num-seqs", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--decode-batch-min-size", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_base_prompts(prompts: list[str] | None) -> list[str]:
    resolved = list(prompts or DEFAULT_PROMPTS)
    if not resolved:
        raise ValueError("at least one prompt is required")
    return resolved


def build_workload_prompts(
    *,
    workload: str,
    base_prompts: list[str],
    canary_batch_size: int,
) -> list[str]:
    if workload == "tiny":
        return list(base_prompts)
    if workload != "canary":
        raise ValueError(f"unsupported workload: {workload!r}")
    if canary_batch_size < 1:
        raise ValueError("canary_batch_size must be >= 1")
    return [base_prompts[index % len(base_prompts)] for index in range(canary_batch_size)]


def find_first_token_divergence(
    variant_outputs: dict[str, list[list[int]]],
    *,
    reference_variant: str = "blockwise",
) -> dict[str, Any] | None:
    if reference_variant not in variant_outputs:
        raise ValueError(f"reference variant missing: {reference_variant}")
    prompt_count = max((len(outputs) for outputs in variant_outputs.values()), default=0)
    max_token_count = 0
    for outputs in variant_outputs.values():
        for token_ids in outputs:
            max_token_count = max(max_token_count, len(token_ids))

    for token_index in range(max_token_count):
        for prompt_index in range(prompt_count):
            tokens: dict[str, int | None] = {}
            for variant, outputs in variant_outputs.items():
                if prompt_index >= len(outputs):
                    tokens[variant] = None
                    continue
                token_ids = outputs[prompt_index]
                tokens[variant] = (
                    int(token_ids[token_index]) if token_index < len(token_ids) else None
                )
            if len(set(tokens.values())) > 1:
                return {
                    "prompt_index": int(prompt_index),
                    "token_index": int(token_index),
                    "generation_step_index": int(token_index),
                    "tokens": tokens,
                    "reference_variant": reference_variant,
                    "reference_token_id": tokens.get(reference_variant),
                }
    return None


def _variant_env(
    variant: str,
    *,
    mosaic_min_decode_batch: int,
    mosaic_throughput_min_decode_batch: int,
    splitk_table_path: str,
) -> dict[str, str]:
    config = VARIANTS[variant]
    env = dict(config.env)
    if variant.startswith("throughput_v2"):
        env["NANOVLLM_JAX_INTERNAL_MOSAIC_MIN_DECODE_BATCH"] = str(
            int(mosaic_min_decode_batch)
        )
        env["NANOVLLM_JAX_INTERNAL_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH"] = str(
            int(mosaic_throughput_min_decode_batch)
        )
        if splitk_table_path:
            env["NANOVLLM_JAX_MOSAIC_THROUGHPUT_SPLITK_TABLE_PATH"] = splitk_table_path
    return env


def _pythonpath_env() -> str:
    repo_src = str(Path(__file__).resolve().parent / "src")
    old = os.environ.get("PYTHONPATH", "")
    return repo_src if not old else f"{repo_src}{os.pathsep}{old}"


def _run_worker(
    *,
    args: argparse.Namespace,
    workload_dir: Path,
    variant: str,
    prompts_path: Path,
    max_num_seqs: int,
    decode_batch_min_size: int,
    save_logits_step: int = -1,
    stop_after_generation_step: int = -1,
    suffix: str = "",
) -> Path:
    variant_dir = workload_dir / f"{variant}{suffix}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--variant",
        variant,
        "--model",
        str(args.model),
        "--output-dir",
        str(variant_dir),
        "--prompts-json",
        str(prompts_path),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--decode-batch-min-size",
        str(decode_batch_min_size),
        "--mosaic-min-decode-batch",
        str(args.mosaic_min_decode_batch),
        "--mosaic-throughput-min-decode-batch",
        str(args.mosaic_throughput_min_decode_batch),
        "--save-logits-step",
        str(save_logits_step),
        "--stop-after-generation-step",
        str(stop_after_generation_step),
    ]
    if args.max_num_batched_tokens:
        command.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.splitk_table_path:
        command.extend(["--splitk-table-path", str(args.splitk_table_path)])
    if args.enforce_eager:
        command.append("--enforce-eager")

    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_env()
    env.update(
        _variant_env(
            variant,
            mosaic_min_decode_batch=args.mosaic_min_decode_batch,
            mosaic_throughput_min_decode_batch=args.mosaic_throughput_min_decode_batch,
            splitk_table_path=str(args.splitk_table_path),
        )
    )
    log_path = variant_dir / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).resolve().parent,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return variant_dir / "worker_summary.json"


def _output_token_ids(summary: dict[str, Any]) -> list[list[int]]:
    return [
        [int(token) for token in record.get("token_ids", [])]
        for record in summary.get("outputs", {}).get("records", [])
    ]


def _record_for_step(records: list[dict[str, Any]], step_index: int) -> dict[str, Any]:
    for record in records:
        if int(record.get("generation_step_index", -1)) == int(step_index):
            return record
    raise ValueError(f"missing generation record for step {step_index}")


def _row_value(record: dict[str, Any], key: str, row: int) -> Any:
    values = record.get(key)
    if isinstance(values, list) and row < len(values):
        return values[row]
    return None


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def summarize_divergent_logits(
    *,
    workload_dir: Path,
    divergence: dict[str, Any],
    tokenizer,
    keep_full_logits: bool,
) -> dict[str, Any]:
    step_index = int(divergence["generation_step_index"])
    prompt_index = int(divergence["prompt_index"])
    records_by_variant: dict[str, dict[str, Any]] = {}
    logits_by_variant: dict[str, np.ndarray] = {}

    for variant in VARIANT_ORDER:
        variant_dir = workload_dir / f"{variant}_logits_step_{step_index}"
        records = _read_jsonl(variant_dir / "generation_debug.jsonl")
        records_by_variant[variant] = _record_for_step(records, step_index)
        logits_path = variant_dir / f"logits_step_{step_index}.npy"
        logits_by_variant[variant] = np.load(logits_path, mmap_mode="r")

    reference_logits = logits_by_variant["blockwise"]
    max_logit_diffs_vs_blockwise: dict[str, dict[str, float]] = {}
    for variant, logits in logits_by_variant.items():
        if variant == "blockwise":
            continue
        abs_diff = np.abs(logits - reference_logits)
        row_abs_diff = np.abs(logits[prompt_index] - reference_logits[prompt_index])
        max_logit_diffs_vs_blockwise[variant] = {
            "all_rows_max_abs_diff": float(np.max(abs_diff)),
            "divergent_row_max_abs_diff": float(np.max(row_abs_diff)),
        }

    variant_details: dict[str, dict[str, Any]] = {}
    for variant, record in records_by_variant.items():
        prefix_token_ids = _row_value(record, "prefix_token_ids", prompt_index) or []
        variant_details[variant] = {
            "sampled_token_id": _row_value(record, "sampled_token_ids", prompt_index),
            "argmax_token_id": _row_value(record, "argmax_token_ids", prompt_index),
            "argmax_logit": _row_value(record, "argmax_logits", prompt_index),
            "argmax_margin": _row_value(record, "argmax_margins", prompt_index),
            "sampled_token_logit": _row_value(record, "sampled_token_logits", prompt_index),
            "sampled_token_rank": _row_value(record, "sampled_token_ranks", prompt_index),
            "top_token_ids": _row_value(record, "top_token_ids", prompt_index),
            "top_logits": _row_value(record, "top_logits", prompt_index),
            "prefix_token_ids": prefix_token_ids,
            "prefix_text": tokenizer.decode(prefix_token_ids, skip_special_tokens=False),
        }

    if not keep_full_logits:
        for variant in VARIANT_ORDER:
            logits_path = (
                workload_dir
                / f"{variant}_logits_step_{step_index}"
                / f"logits_step_{step_index}.npy"
            )
            try:
                logits_path.unlink()
            except FileNotFoundError:
                pass

    return {
        **divergence,
        "max_logit_diffs_vs_blockwise": max_logit_diffs_vs_blockwise,
        "variant_details": variant_details,
    }


def _run_workload(args: argparse.Namespace, workload: str, base_prompts: list[str]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    workload_dir = output_dir / workload
    workload_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_workload_prompts(
        workload=workload,
        base_prompts=base_prompts,
        canary_batch_size=args.canary_batch_size,
    )
    prompts_path = workload_dir / "prompts.json"
    _write_json(prompts_path, {"prompts": prompts})

    max_num_seqs = len(prompts)
    decode_batch_min_size = len(prompts) if workload == "canary" else 1
    worker_summaries: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        summary_path = _run_worker(
            args=args,
            workload_dir=workload_dir,
            variant=variant,
            prompts_path=prompts_path,
            max_num_seqs=max_num_seqs,
            decode_batch_min_size=decode_batch_min_size,
        )
        worker_summaries[variant] = _read_json(summary_path)

    variant_outputs = {
        variant: _output_token_ids(summary)
        for variant, summary in worker_summaries.items()
    }
    divergence = find_first_token_divergence(variant_outputs)
    if divergence is not None:
        step_index = int(divergence["generation_step_index"])
        for variant in VARIANT_ORDER:
            _run_worker(
                args=args,
                workload_dir=workload_dir,
                variant=variant,
                prompts_path=prompts_path,
                max_num_seqs=max_num_seqs,
                decode_batch_min_size=decode_batch_min_size,
                save_logits_step=step_index,
                stop_after_generation_step=step_index,
                suffix=f"_logits_step_{step_index}",
            )
        tokenizer = _load_tokenizer(str(args.model))
        divergence = summarize_divergent_logits(
            workload_dir=workload_dir,
            divergence=divergence,
            tokenizer=tokenizer,
            keep_full_logits=bool(args.keep_full_logits),
        )
        _write_json(workload_dir / "first_divergence.json", divergence)

    workload_summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": workload,
        "prompt_count": len(prompts),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "variants": list(VARIANT_ORDER),
        "worker_summaries": {
            variant: str(workload_dir / variant / "worker_summary.json")
            for variant in VARIANT_ORDER
        },
        "first_divergence": divergence,
        "outputs_match_all_variants": divergence is None,
    }
    _write_json(workload_dir / "summary.json", workload_summary)
    return workload_summary


def _worker_main(args: argparse.Namespace) -> None:
    if args.variant is None:
        raise ValueError("--variant is required in worker mode")
    if args.prompts_json is None:
        raise ValueError("--prompts-json is required in worker mode")

    variant = args.variant
    variant_config = VARIANTS[variant]
    os.environ.update(
        _variant_env(
            variant,
            mosaic_min_decode_batch=args.mosaic_min_decode_batch,
            mosaic_throughput_min_decode_batch=args.mosaic_throughput_min_decode_batch,
            splitk_table_path=str(args.splitk_table_path),
        )
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "generation_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()

    prompts_payload = _read_json(Path(args.prompts_json).expanduser())
    prompts = [str(prompt) for prompt in prompts_payload.get("prompts", [])]
    if not prompts:
        raise ValueError("worker prompt file has no prompts")

    from nanovllm_jax import LLM, SamplingParams

    llm_kwargs = {
        "max_num_seqs": int(args.max_num_seqs or len(prompts)),
        "max_model_len": int(args.max_model_len),
        "decode_batch_min_size": int(args.decode_batch_min_size),
        "decode_attention_backend": variant_config.decode_attention_backend,
        "enforce_eager": bool(args.enforce_eager),
        "tensor_parallel_size": 1,
    }
    if args.max_num_batched_tokens:
        llm_kwargs["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
    else:
        # Keep canary prefill in one batch when prompts are short enough.
        llm_kwargs["max_num_batched_tokens"] = max(
            int(args.max_model_len),
            int(args.max_model_len) * int(args.max_num_seqs or len(prompts)),
        )

    llm = LLM(str(args.model), **llm_kwargs)
    llm.model_runner.capture_generation_debug = True
    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
    )
    tokenized = llm.tokenizer(
        list(prompts),
        add_special_tokens=True,
        return_attention_mask=False,
    )
    for prompt_ids in tokenized["input_ids"]:
        llm.add_request(prompt_ids, sampling_params)

    outputs: dict[int, list[int]] = {}
    generation_step_index = 0
    stopped_early = False
    while not llm.is_finished():
        step_outputs, num_tokens = llm.step()
        debug = llm.model_runner.consume_last_generation_debug(
            top_k=int(args.top_k),
            save_logits_path=(
                output_dir / f"logits_step_{generation_step_index}.npy"
                if int(args.save_logits_step) == generation_step_index
                else None
            ),
        )
        if debug is not None:
            debug.update(
                {
                    "variant": variant,
                    "generation_step_index": int(generation_step_index),
                    "num_tokens_delta": int(num_tokens),
                }
            )
            _append_jsonl(debug_path, debug)
            if int(args.stop_after_generation_step) == generation_step_index:
                stopped_early = True
                break
            generation_step_index += 1

        for seq_id, token_ids in step_outputs:
            outputs[int(seq_id)] = [int(token) for token in token_ids]

    if not stopped_early:
        output_token_ids = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        decoded_texts = llm.tokenizer.batch_decode(
            output_token_ids,
            skip_special_tokens=True,
        )
        output_records = [
            {"index": index, "text": text, "token_ids": token_ids}
            for index, (text, token_ids) in enumerate(zip(decoded_texts, output_token_ids))
        ]
    else:
        output_records = []

    if hasattr(llm, "exit"):
        llm.exit()

    summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "model_path": str(args.model),
        "output_dir": str(output_dir),
        "generation_debug_path": str(debug_path),
        "stopped_early": bool(stopped_early),
        "runtime": {
            "decode_attention_backend": variant_config.decode_attention_backend,
            "env": _variant_env(
                variant,
                mosaic_min_decode_batch=args.mosaic_min_decode_batch,
                mosaic_throughput_min_decode_batch=args.mosaic_throughput_min_decode_batch,
                splitk_table_path=str(args.splitk_table_path),
            ),
            "max_num_seqs": int(llm_kwargs["max_num_seqs"]),
            "decode_batch_min_size": int(llm_kwargs["decode_batch_min_size"]),
            "max_num_batched_tokens": int(llm_kwargs["max_num_batched_tokens"]),
            "max_model_len": int(llm_kwargs["max_model_len"]),
            "enforce_eager": bool(llm_kwargs["enforce_eager"]),
        },
        "sampling": {
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
        },
        "prompts": {
            "count": len(prompts),
            "items": prompts,
        },
        "outputs": {
            "count": len(output_records),
            "token_counts": [
                len(record.get("token_ids", [])) for record in output_records
            ],
            "records": output_records,
        },
    }
    _write_json(output_dir / "worker_summary.json", summary)
    print(output_dir / "worker_summary.json")


def main() -> None:
    args = _parse_args()
    if args.worker:
        _worker_main(args)
        return

    base_prompts = _resolve_base_prompts(args.prompts)
    workloads = ["tiny", "canary"] if args.workload == "both" else [args.workload]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    workload_summaries = {
        workload: _run_workload(args, workload, base_prompts)
        for workload in workloads
    }
    final_summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model),
        "workloads": workload_summaries,
    }
    _write_json(output_dir / "summary.json", final_summary)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
