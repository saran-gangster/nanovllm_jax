from __future__ import annotations

import json
from pathlib import Path

from nanovllm_jax.utils.decode_kernel_bench import (
    build_worker_command,
    compare_kernel_benchmark_summaries,
    extract_case_env,
    parse_case_spec,
    summarize_kernel_case_runs,
)


def test_parse_case_spec_requires_name_and_family() -> None:
    case = parse_case_spec("name=baseline,family=baseline,block_q=64,use_schedule_barrier=false")

    assert case["name"] == "baseline"
    assert case["family"] == "baseline"
    assert case["block_q"] == 64
    assert case["use_schedule_barrier"] is False


def test_parse_case_spec_keeps_numeric_values_numeric() -> None:
    case = parse_case_spec(
        "name=throughput_v2,family=throughput_v2,num_compute_wgs=1,throughput_split_k=0",
    )

    assert case["num_compute_wgs"] == 1
    assert isinstance(case["num_compute_wgs"], int)
    assert case["throughput_split_k"] == 0
    assert isinstance(case["throughput_split_k"], int)


def test_build_worker_command_includes_case_and_common_args(tmp_path: Path) -> None:
    command = build_worker_command(
        repo_root="/repo",
        common_args={
            "batch_size": 512,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 256,
            "max_blocks_per_seq": 16,
            "num_blocks": 4096,
            "dtype": "bfloat16",
            "seed": 0,
        },
        case={
            "name": "baseline",
            "family": "baseline",
            "block_q": 64,
            "block_kv": 64,
            "num_compute_wgs": 1,
        },
        output_json=tmp_path / "out.json",
        warmup=5,
        iters=20,
        verify_against_blockwise=True,
    )

    assert command[:2]
    assert "--family" in command
    assert "baseline" in command
    assert "--verify-against-blockwise" in command
    assert "--output-json" in command


def test_build_worker_command_emits_boolean_optional_false_flag(tmp_path: Path) -> None:
    command = build_worker_command(
        repo_root="/repo",
        common_args={
            "batch_size": 512,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 256,
            "max_blocks_per_seq": 16,
            "num_blocks": 4096,
            "dtype": "bfloat16",
            "seed": 0,
        },
        case={
            "name": "throughput_v2",
            "family": "throughput_v2",
            "block_q": 64,
            "block_kv": 64,
            "num_compute_wgs": 1,
            "use_schedule_barrier": False,
        },
        output_json=tmp_path / "out.json",
        warmup=5,
        iters=20,
    )

    assert "--no-use-schedule-barrier" in command
    assert "--use-schedule-barrier" not in command


def test_extract_case_env_and_omit_from_worker_command(tmp_path: Path) -> None:
    case = {
        "name": "throughput_v2_mosaic",
        "family": "throughput_v2",
        "env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": 1,
        "env__NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
        "block_q": 64,
    }

    env = extract_case_env(case)
    command = build_worker_command(
        repo_root="/repo",
        common_args={
            "batch_size": 512,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 256,
            "max_blocks_per_seq": 16,
            "num_blocks": 4096,
            "dtype": "bfloat16",
            "seed": 0,
        },
        case=case,
        output_json=tmp_path / "out.json",
        warmup=5,
        iters=20,
    )

    assert env == {
        "NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "1",
        "NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
    }
    assert "--env--nanovllm-jax-enable-throughput-v2-mosaic" not in command
    assert "--env--nanovllm-jax-mosaic-decode-kernel" not in command


def test_summarize_and_compare_kernel_case_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a.json"
    run_b = tmp_path / "run_b.json"
    run_a.write_text(
        json.dumps(
            {
                "family": "baseline",
                "shape": {"batch_size": 512},
                "dtype": "bfloat16",
                "timings": {
                    "mean_ms": 10.0,
                    "p50_ms": 9.5,
                    "min_ms": 9.0,
                    "max_ms": 11.0,
                    "compile_and_first_run_s": 1.0,
                },
                "output": {"checksum_f32_sum": 1.25},
                "family_notes": {"note": "x"},
                "verify": None,
            }
        ),
        encoding="utf-8",
    )
    run_b.write_text(
        json.dumps(
            {
                "family": "baseline",
                "shape": {"batch_size": 512},
                "dtype": "bfloat16",
                "timings": {
                    "mean_ms": 12.0,
                    "p50_ms": 11.5,
                    "min_ms": 10.5,
                    "max_ms": 12.5,
                    "compile_and_first_run_s": 1.5,
                },
                "output": {"checksum_f32_sum": 1.25},
                "family_notes": {"note": "x"},
                "verify": None,
            }
        ),
        encoding="utf-8",
    )

    baseline_summary = summarize_kernel_case_runs(
        [run_a],
        case_name="baseline",
        pass_name="quick",
    )
    candidate_summary = summarize_kernel_case_runs(
        [run_b],
        case_name="candidate",
        pass_name="quick",
    )

    before_path = tmp_path / "baseline_summary.json"
    after_path = tmp_path / "candidate_summary.json"
    before_path.write_text(json.dumps(baseline_summary), encoding="utf-8")
    after_path.write_text(json.dumps(candidate_summary), encoding="utf-8")

    comparison = compare_kernel_benchmark_summaries(before_path, after_path)

    assert baseline_summary["timings"]["mean_of_means_ms"] == 10.0
    assert candidate_summary["timings"]["mean_of_means_ms"] == 12.0
    assert comparison["timings"]["mean_of_means_ms"]["delta"] == 2.0
