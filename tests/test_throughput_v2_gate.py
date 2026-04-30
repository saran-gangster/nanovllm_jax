from __future__ import annotations

import pytest

from nanovllm_jax.utils.throughput_v2_gate import (
    DEFAULT_RUNTIME_GATE_PROMPTS,
    DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES,
    build_canary_kernel_table,
    build_promotion_gate_cases,
    build_promotion_gate_rows,
    build_speed_window_split_rows,
    build_splitk_override_table,
    build_split_sweep_cases,
    merge_splitk_override_tables,
    summarize_runtime_gate,
)


def test_build_promotion_gate_rows_extended_includes_expert_rows() -> None:
    rows = build_promotion_gate_rows("extended")

    assert [row.shape_key for row in rows] == [
        "b512_hd128_mb16_bs256",
        "b512_hd128_mb32_bs256",
        "b512_hd128_mb64_bs256",
        "b1024_hd128_mb16_bs256",
        "b2048_hd128_mb16_bs256",
        "b4096_hd128_mb16_bs256",
        "b512_hd128_mb24_bs256",
        "b512_hd128_mb48_bs256",
        "b1024_hd128_mb32_bs256",
        "b2048_hd128_mb32_bs256",
    ]


def test_build_promotion_gate_cases_supports_mosaic_and_jax_variants() -> None:
    cases = build_promotion_gate_cases(include_jax_reference=True)

    assert cases[0]["name"] == "blockwise"
    assert cases[1]["name"] == "throughput_v2_mosaic"
    assert cases[1]["env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC"] == "1"
    assert cases[2]["name"] == "throughput_v2_jax"
    assert cases[2]["env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC"] == "0"


def test_build_canary_kernel_table_uses_extended_rows_by_default() -> None:
    table = build_canary_kernel_table()

    assert len(table) == 10
    assert table[
        "batch=512,head_dim=128,blocks=16,block_size=256,num_heads=16,num_kv_heads=8,dtype=bfloat16"
    ] == "throughput_v2"
    assert table[
        "batch=512,head_dim=128,blocks=24,block_size=256,num_heads=16,num_kv_heads=8,dtype=bfloat16"
    ] == "throughput_v2"
    assert table[
        "batch=512,head_dim=128,blocks=48,block_size=256,num_heads=16,num_kv_heads=8,dtype=bfloat16"
    ] == "throughput_v2"
    assert table[
        "batch=2048,head_dim=128,blocks=32,block_size=256,num_heads=16,num_kv_heads=8,dtype=bfloat16"
    ] == "throughput_v2"


def test_build_split_sweep_cases_emits_requested_candidates() -> None:
    cases = build_split_sweep_cases(split_candidates=(2, 4, 8))

    assert [case["name"] for case in cases] == [
        "blockwise",
        "throughput_v2_mosaic_split2",
        "throughput_v2_mosaic_split4",
        "throughput_v2_mosaic_split8",
    ]
    assert cases[1]["throughput_split_k"] == 2
    assert cases[3]["throughput_split_k"] == 8


def test_build_split_sweep_cases_can_compare_current_mosaic_default() -> None:
    cases = build_split_sweep_cases(
        split_candidates=(1, 2),
        include_current_mosaic=True,
    )

    assert [case["name"] for case in cases] == [
        "blockwise",
        "throughput_v2_mosaic_default",
        "throughput_v2_mosaic_split1",
        "throughput_v2_mosaic_split2",
    ]
    assert cases[1]["env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC"] == "1"
    assert "throughput_split_k" not in cases[1]


def test_build_speed_window_split_rows_matches_expert_target_rows() -> None:
    rows = build_speed_window_split_rows()

    assert DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES == (1, 2, 4, 8, 16)
    assert [row.shape_key for row in rows] == [
        "b512_hd128_mb24_bs256",
        "b512_hd128_mb48_bs256",
        "b512_hd128_mb64_bs256",
        "b1024_hd128_mb32_bs256",
        "b2048_hd128_mb32_bs256",
    ]


def test_build_splitk_override_table_targets_default_long_context_row() -> None:
    table = build_splitk_override_table(split_k=4)

    assert table == {
        "batch=512,head_dim=128,blocks=64,block_size=256,num_heads=16,num_kv_heads=8,dtype=bfloat16": 4,
    }


def test_merge_splitk_override_tables_rejects_conflicts() -> None:
    table_a = {
        "batch=512,head_dim=128,blocks=64,block_size=256,num_heads=16,num_kv_heads=8,dtype=bfloat16": 4,
    }
    table_b = {
        "batch=1024,head_dim=128,blocks=32,block_size=256,num_heads=16,num_kv_heads=8,dtype=bfloat16": 8,
    }

    merged = merge_splitk_override_tables([table_a, table_b])

    assert merged == {**table_a, **table_b}
    with pytest.raises(ValueError):
        merge_splitk_override_tables([table_a, {next(iter(table_a)): 8}])


def test_runtime_gate_defaults_are_three_deterministic_prompts() -> None:
    assert len(DEFAULT_RUNTIME_GATE_PROMPTS) == 3
    assert all(prompt.endswith("sentence.") for prompt in DEFAULT_RUNTIME_GATE_PROMPTS)


def test_summarize_runtime_gate_compares_outputs_and_timings() -> None:
    blockwise = {
        "summary_path": "/tmp/blockwise.json",
        "outputs": {"records": [{"token_ids": [1, 2, 3], "text": "x"}]},
        "timings": {
            "top_level_total_s": 5.0,
            "model_execute_total_s": 4.5,
        },
    }
    throughput_v2 = {
        "summary_path": "/tmp/throughput_v2.json",
        "outputs": {"records": [{"token_ids": [1, 2, 3], "text": "x"}]},
        "timings": {
            "top_level_total_s": 4.6,
            "model_execute_total_s": 4.2,
        },
    }

    summary = summarize_runtime_gate(
        blockwise_summary=blockwise,
        throughput_v2_summary=throughput_v2,
    )

    assert summary["outputs_match"] is True
    assert summary["timings"]["top_level_total_s"]["delta"] == pytest.approx(-0.4)
    assert summary["timings"]["model_execute_total_s"]["delta"] == pytest.approx(-0.3)
