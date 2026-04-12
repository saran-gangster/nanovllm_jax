#!/usr/bin/env python3
"""Benchmark direct decode-attention families on synthetic paged-KV inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.layers.mosaic_gpu_attention import (
    MosaicAttentionConfig,
    _select_latency_k_splits,
    _select_throughput_k_splits,
    batched_decode_attention_mosaic,
    build_paged_decode_throughput_v2_plan,
    paged_decode_attention_mosaic_latency,
    paged_decode_attention_mosaic_throughput,
    paged_decode_attention_mosaic_throughput_v2,
    prepare_decode_metadata,
)
from nanovllm_jax.layers.paged_attention import paged_decode_attention_blockwise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark direct decode-attention families on synthetic paged-KV inputs.",
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=("blockwise", "baseline", "latency", "throughput", "throughput_v2"),
        help="Decode family to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--max-blocks-per-seq",
        type=int,
        default=16,
        help="Logical context length in KV-cache pages per sequence.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4096,
        help="Physical KV-cache page pool size.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-json", help="Optional JSON output path.")
    parser.add_argument("--block-q", type=int, default=64)
    parser.add_argument("--block-kv", type=int, default=64)
    parser.add_argument("--max-concurrent-steps", type=int, default=2)
    parser.add_argument(
        "--num-compute-wgs",
        type=int,
        default=2,
        choices=(1, 2),
    )
    parser.add_argument(
        "--use-schedule-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--throughput-split-k",
        type=int,
        default=0,
        help="Optional throughput split-k override. 0 keeps heuristic selection.",
    )
    parser.add_argument(
        "--verify-against-blockwise",
        action="store_true",
        help="Compute a one-off max-abs diff against the blockwise reference.",
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> jnp.dtype:
    return {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }[name]


def _block_until_ready(tree: Any) -> Any:
    return jax.tree_util.tree_map(jax.block_until_ready, tree)


def _shape_list(value: Any) -> list[int]:
    return list(getattr(value, "shape", ()))


def _collect_git_info() -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                list(args),
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parent,
            )
        except Exception:
            return None
        output = result.stdout.strip()
        return output or None

    return {
        "sha": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain") or ""),
    }


def _make_inputs(args: argparse.Namespace, dtype: jnp.dtype):
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")
    if args.num_blocks < args.max_blocks_per_seq:
        raise ValueError("--num-blocks must be >= --max-blocks-per-seq")

    key = jax.random.PRNGKey(args.seed)
    q_key, k_key, v_key, table_key = jax.random.split(key, 4)
    q = jax.random.normal(
        q_key,
        (args.batch_size, args.num_heads, args.head_dim),
        dtype=dtype,
    )
    k_cache = jax.random.normal(
        k_key,
        (args.num_blocks, args.block_size, args.num_kv_heads, args.head_dim),
        dtype=dtype,
    )
    v_cache = jax.random.normal(
        v_key,
        (args.num_blocks, args.block_size, args.num_kv_heads, args.head_dim),
        dtype=dtype,
    )
    max_start = max(args.num_blocks - args.max_blocks_per_seq + 1, 1)
    row_starts = jax.random.randint(
        table_key,
        (args.batch_size, 1),
        minval=0,
        maxval=max_start,
        dtype=jnp.int32,
    )
    block_tables = (
        row_starts
        + jnp.arange(args.max_blocks_per_seq, dtype=jnp.int32)[None, :]
    ) % args.num_blocks
    context_lens = jnp.full(
        (args.batch_size,),
        args.max_blocks_per_seq * args.block_size,
        dtype=jnp.int32,
    )
    scale = float(1.0 / math.sqrt(args.head_dim))
    return q, k_cache, v_cache, block_tables, context_lens, scale


def _config_from_args(args: argparse.Namespace) -> MosaicAttentionConfig:
    return MosaicAttentionConfig(
        block_q=args.block_q,
        block_kv=args.block_kv,
        max_concurrent_steps=args.max_concurrent_steps,
        use_schedule_barrier=args.use_schedule_barrier,
        num_compute_wgs=args.num_compute_wgs,
    )


def _build_family_runner(
    *,
    args: argparse.Namespace,
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
) -> tuple[callable[[], Any], dict[str, Any]]:
    family = args.family
    notes: dict[str, Any] = {"family": family}

    if family == "blockwise":
        return (
            lambda: paged_decode_attention_blockwise(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                scale,
                args.block_size,
            ),
            notes,
        )

    config = _config_from_args(args)
    notes["config"] = {
        "block_q": config.block_q,
        "block_kv": config.block_kv,
        "max_concurrent_steps": config.max_concurrent_steps,
        "use_schedule_barrier": config.use_schedule_barrier,
        "num_compute_wgs": config.num_compute_wgs,
    }

    if family == "baseline":
        metadata_start = time.perf_counter()
        metadata = prepare_decode_metadata(
            block_tables,
            context_lens,
            args.batch_size,
            config.block_q,
            args.block_size,
            config.block_kv,
            include_unused_fields=False,
        )
        _block_until_ready(metadata)
        notes["metadata_prepare_s"] = time.perf_counter() - metadata_start
        notes["metadata_shape"] = {
            "tile_chunk_row_indices": _shape_list(metadata.tile_chunk_row_indices),
            "tile_chunk_counts": _shape_list(metadata.tile_chunk_counts),
        }
        return (
            lambda: batched_decode_attention_mosaic(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                scale=scale,
                config=config,
                metadata=metadata,
            ),
            notes,
        )

    if family == "latency":
        notes["expected_k_splits"] = _select_latency_k_splits(
            args.max_blocks_per_seq,
            args.head_dim,
        )
        prepared_metadata_cache: dict[tuple[Any, ...], object] = {}
        notes["prepared_metadata_cache_primed"] = False

        def _run_latency():
            return paged_decode_attention_mosaic_latency(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                scale=scale,
                block_size=args.block_size,
                config=config,
                prepared_metadata_cache=prepared_metadata_cache,
            )

        notes["_cache"] = prepared_metadata_cache
        return _run_latency, notes

    if family == "throughput":
        notes["expected_k_splits"] = _select_throughput_k_splits(
            split_k=args.throughput_split_k,
            batch_size=args.batch_size,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            block_q=config.block_q,
            max_blocks_per_seq=args.max_blocks_per_seq,
            block_size=args.block_size,
            block_kv=config.block_kv,
        )
        prepared_metadata_cache: dict[tuple[Any, ...], object] = {}
        notes["prepared_metadata_cache_primed"] = False

        def _run_throughput():
            return paged_decode_attention_mosaic_throughput(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                scale=scale,
                block_size=args.block_size,
                config=config,
                split_k=args.throughput_split_k,
                prepared_metadata_cache=prepared_metadata_cache,
            )

        notes["_cache"] = prepared_metadata_cache
        return _run_throughput, notes

    if family == "throughput_v2":
        notes["expected_k_splits"] = _select_throughput_k_splits(
            split_k=args.throughput_split_k,
            batch_size=args.batch_size,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            block_q=config.block_q,
            max_blocks_per_seq=args.max_blocks_per_seq,
            block_size=args.block_size,
            block_kv=config.block_kv,
        )
        plan = build_paged_decode_throughput_v2_plan(
            q=q,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=args.block_size,
            config=config,
            split_k=args.throughput_split_k,
        )
        notes["v2_plan"] = {
            "block_q": plan.block_q,
            "block_kv": plan.block_kv,
            "k_splits": plan.k_splits,
            "pages_per_partition": plan.pages_per_partition,
            "uses_wrapper_partitioning": plan.uses_wrapper_partitioning,
            "reduction_boundary": plan.reduction_boundary,
        }
        prepared_metadata_cache: dict[tuple[Any, ...], object] = {}
        notes["prepared_metadata_cache_primed"] = False

        def _run_throughput_v2():
            return paged_decode_attention_mosaic_throughput_v2(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                scale=scale,
                block_size=args.block_size,
                config=config,
                split_k=args.throughput_split_k,
                prepared_metadata_cache=prepared_metadata_cache,
            )

        notes["_cache"] = prepared_metadata_cache
        return _run_throughput_v2, notes

    raise ValueError(f"Unsupported family: {family}")


def _prime_family_cache(notes: dict[str, Any]) -> None:
    cache = notes.get("_cache")
    if isinstance(cache, dict):
        notes["prepared_metadata_cache_entries"] = len(cache)
        notes["prepared_metadata_cache_primed"] = bool(cache)
        notes.pop("_cache", None)


def _output_checksum(output: Any) -> float:
    leaves = jax.tree_util.tree_leaves(output)
    if not leaves:
        return 0.0
    checksum = 0.0
    for leaf in leaves:
        checksum += float(jnp.asarray(leaf, dtype=jnp.float32).sum())
    return checksum


def _max_abs_diff(a: Any, b: Any) -> float:
    a_leaves = jax.tree_util.tree_leaves(a)
    b_leaves = jax.tree_util.tree_leaves(b)
    max_diff = 0.0
    for left, right in zip(a_leaves, b_leaves, strict=True):
        diff = jnp.max(jnp.abs(jnp.asarray(left, dtype=jnp.float32) - jnp.asarray(right, dtype=jnp.float32)))
        max_diff = max(max_diff, float(diff))
    return max_diff


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)
    q, k_cache, v_cache, block_tables, context_lens, scale = _make_inputs(args, dtype)
    run_family, family_notes = _build_family_runner(
        args=args,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
    )

    compile_start = time.perf_counter()
    first_output = run_family()
    _block_until_ready(first_output)
    compile_and_first_run_s = time.perf_counter() - compile_start
    _prime_family_cache(family_notes)

    warmup_times_s: list[float] = []
    for _ in range(args.warmup):
        start = time.perf_counter()
        output = run_family()
        _block_until_ready(output)
        warmup_times_s.append(time.perf_counter() - start)

    iter_times_s: list[float] = []
    last_output = None
    for _ in range(args.iters):
        start = time.perf_counter()
        last_output = run_family()
        _block_until_ready(last_output)
        iter_times_s.append(time.perf_counter() - start)

    verify = None
    if args.verify_against_blockwise and args.family != "blockwise":
        ref = paged_decode_attention_blockwise(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            scale,
            args.block_size,
        )
        _block_until_ready(ref)
        verify = {
            "reference_family": "blockwise",
            "max_abs_diff": _max_abs_diff(last_output if last_output is not None else first_output, ref),
        }

    mean_s = statistics.mean(iter_times_s)
    p50_s = statistics.median(iter_times_s)
    result = {
        "format_version": 1,
        "family": args.family,
        "shape": {
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "max_blocks_per_seq": args.max_blocks_per_seq,
            "num_blocks": args.num_blocks,
        },
        "dtype": args.dtype,
        "scale": scale,
        "iterations": {
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "timings": {
            "compile_and_first_run_s": compile_and_first_run_s,
            "warmup_mean_ms": statistics.mean(warmup_times_s) * 1000.0 if warmup_times_s else None,
            "mean_ms": mean_s * 1000.0,
            "p50_ms": p50_s * 1000.0,
            "min_ms": min(iter_times_s) * 1000.0,
            "max_ms": max(iter_times_s) * 1000.0,
            "iter_times_ms": [value * 1000.0 for value in iter_times_s],
        },
        "output": {
            "checksum_f32_sum": _output_checksum(last_output if last_output is not None else first_output),
        },
        "verify": verify,
        "family_notes": family_notes,
        "runtime": {
            "devices": [str(device) for device in jax.devices()],
            "platform": jax.default_backend(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
        },
        "git": _collect_git_info(),
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(output_path)
    else:
        print(payload)


if __name__ == "__main__":
    main()
