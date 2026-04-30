from __future__ import annotations

from pathlib import Path

import bench_throughput_v2_speed_window as speed_window
from nanovllm_jax.utils.throughput_v2_gate import PromotionGateRow


def test_parse_split_rows_uses_speed_window_defaults() -> None:
    rows = speed_window._parse_split_rows(
        "speed_window",
        block_size=256,
        num_heads=16,
        num_kv_heads=8,
        dtype="bfloat16",
    )

    assert [row.shape_key for row in rows] == [
        "b512_hd128_mb24_bs256",
        "b512_hd128_mb48_bs256",
        "b512_hd128_mb64_bs256",
        "b1024_hd128_mb32_bs256",
        "b2048_hd128_mb32_bs256",
    ]
    assert all(row.num_heads == 16 for row in rows)
    assert all(row.num_kv_heads == 8 for row in rows)


def test_parse_split_rows_accepts_explicit_specs() -> None:
    rows = speed_window._parse_split_rows(
        "512x128x64,1024:128:32:256",
        block_size=256,
        num_heads=16,
        num_kv_heads=8,
        dtype="bfloat16",
    )

    assert [row.shape_key for row in rows] == [
        "b512_hd128_mb64_bs256",
        "b1024_hd128_mb32_bs256",
    ]


def test_split_sweep_command_includes_current_mosaic_and_candidates(tmp_path: Path) -> None:
    command = speed_window._split_sweep_command(
        repo_root=Path("/repo"),
        output_dir=tmp_path,
        row=PromotionGateRow(512, 128, 64),
        split_candidates=(1, 2, 4, 8, 16),
        warmup=5,
        iters=20,
        repetitions=1,
        num_blocks=4096,
        seed=0,
        include_jax_reference=False,
        verify_against_blockwise=True,
    )

    assert str(Path("/repo") / "bench_throughput_v2_split_sweep.py") in command
    assert "--include-current-mosaic" in command
    assert "--split-candidates" in command
    assert "1,2,4,8,16" in command
    assert "--verify-against-blockwise" in command


def test_runtime_gate_command_targets_profile_script(tmp_path: Path) -> None:
    command = speed_window._runtime_gate_command(
        repo_root=Path("/repo"),
        output_dir=tmp_path,
        model="/workspace/models/Qwen3-0.6B",
        max_tokens=64,
    )

    assert str(Path("/repo") / "profile_throughput_v2_runtime_gate.py") in command
    assert "--model" in command
    assert "/workspace/models/Qwen3-0.6B" in command
    assert "--max-tokens" in command


def test_winner_improvement_reports_current_default_speedup() -> None:
    summary = {
        "winner": {
            "qualified": True,
            "split_k": 8,
            "mean_of_means_ms": 9.0,
        },
        "gate": {
            "blockwise_mean_of_means_ms": 12.0,
            "current_mosaic_default_mean_of_means_ms": 10.0,
        },
    }

    improvement = speed_window._winner_improvement(summary)

    assert improvement is not None
    assert improvement["split_k"] == 8
    assert improvement["speedup_vs_blockwise_pct"] == 25.0
    assert improvement["speedup_vs_current_mosaic_default_pct"] == 10.0
