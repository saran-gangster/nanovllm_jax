"""Shared throughput-v2 promotion gate definitions."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class PromotionGateRow:
    batch_size: int
    head_dim: int
    max_blocks_per_seq: int
    block_size: int = 256
    num_heads: int = 16
    num_kv_heads: int = 8
    dtype: str = "bfloat16"

    @property
    def shape_key(self) -> str:
        return (
            f"b{self.batch_size}_hd{self.head_dim}_mb{self.max_blocks_per_seq}"
            f"_bs{self.block_size}"
        )

    @property
    def table_key(self) -> str:
        return (
            f"batch={self.batch_size},head_dim={self.head_dim},"
            f"blocks={self.max_blocks_per_seq},block_size={self.block_size},"
            f"num_heads={self.num_heads},num_kv_heads={self.num_kv_heads},"
            f"dtype={self.dtype}"
        )

    @property
    def legacy_table_key(self) -> str:
        return (
            f"batch={self.batch_size},head_dim={self.head_dim},"
            f"blocks={self.max_blocks_per_seq},block_size={self.block_size}"
        )


PRIMARY_PROMOTION_ROWS: tuple[PromotionGateRow, ...] = (
    PromotionGateRow(512, 128, 16),
    PromotionGateRow(512, 128, 32),
    PromotionGateRow(512, 128, 64),
    PromotionGateRow(1024, 128, 16),
    PromotionGateRow(2048, 128, 16),
    PromotionGateRow(4096, 128, 16),
)

DEFAULT_SPLIT_SWEEP_ROW = PromotionGateRow(512, 128, 64)

SPEED_WINDOW_SPLIT_SWEEP_ROWS: tuple[PromotionGateRow, ...] = (
    PromotionGateRow(512, 128, 24),
    PromotionGateRow(512, 128, 48),
    PromotionGateRow(512, 128, 64),
    PromotionGateRow(1024, 128, 32),
    PromotionGateRow(2048, 128, 32),
)

DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8, 16)

EXTENDED_PROMOTION_ROWS: tuple[PromotionGateRow, ...] = (
    *PRIMARY_PROMOTION_ROWS,
    PromotionGateRow(512, 128, 24),
    PromotionGateRow(512, 128, 48),
    PromotionGateRow(1024, 128, 32),
    PromotionGateRow(2048, 128, 32),
)

DEFAULT_RUNTIME_GATE_PROMPTS: tuple[str, ...] = (
    "Summarize the word latency in one short sentence.",
    "Summarize the phrase throughput optimization in one short sentence.",
    "Summarize the phrase decode kernel in one short sentence.",
)


def build_promotion_gate_rows(matrix: str = "extended") -> list[PromotionGateRow]:
    matrix_key = str(matrix).strip().lower()
    if matrix_key == "primary":
        return list(PRIMARY_PROMOTION_ROWS)
    if matrix_key == "extended":
        return list(EXTENDED_PROMOTION_ROWS)
    raise ValueError(f"Unsupported promotion matrix: {matrix!r}")


def build_speed_window_split_rows() -> list[PromotionGateRow]:
    """Rows where split policy is most likely to move throughput-v2."""
    return list(SPEED_WINDOW_SPLIT_SWEEP_ROWS)


def build_canary_kernel_table(
    rows: list[PromotionGateRow] | tuple[PromotionGateRow, ...] | None = None,
    *,
    family: str = "throughput_v2",
) -> dict[str, str]:
    resolved_rows = list(EXTENDED_PROMOTION_ROWS if rows is None else rows)
    return {row.table_key: str(family) for row in resolved_rows}


def build_promotion_gate_cases(*, include_jax_reference: bool = False) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "name": "blockwise",
            "family": "blockwise",
        },
        {
            "name": "throughput_v2_mosaic",
            "family": "throughput_v2",
            "env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "1",
            "env__NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
        },
    ]
    if include_jax_reference:
        cases.append(
            {
                "name": "throughput_v2_jax",
                "family": "throughput_v2",
                "env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "0",
                "env__NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
            }
        )
    return cases


def build_split_sweep_cases(
    *,
    split_candidates: tuple[int, ...] = DEFAULT_SPEED_WINDOW_SPLIT_CANDIDATES,
    include_current_mosaic: bool = False,
    include_jax_reference: bool = False,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "name": "blockwise",
            "family": "blockwise",
        },
    ]
    if include_current_mosaic:
        cases.append(
            {
                "name": "throughput_v2_mosaic_default",
                "family": "throughput_v2",
                "env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "1",
                "env__NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
            }
        )
    if include_jax_reference:
        cases.append(
            {
                "name": "throughput_v2_jax",
                "family": "throughput_v2",
                "env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "0",
                "env__NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
            }
        )
    for split_k in split_candidates:
        cases.append(
            {
                "name": f"throughput_v2_mosaic_split{split_k}",
                "family": "throughput_v2",
                "throughput_split_k": int(split_k),
                "env__NANOVLLM_JAX_ENABLE_THROUGHPUT_V2_MOSAIC": "1",
                "env__NANOVLLM_JAX_MOSAIC_DECODE_KERNEL": "throughput_v2",
            }
        )
    return cases


def build_splitk_override_table(
    *,
    split_k: int,
    row: PromotionGateRow = DEFAULT_SPLIT_SWEEP_ROW,
) -> dict[str, int]:
    if int(split_k) < 1:
        raise ValueError("split_k override must be >= 1")
    return {row.table_key: int(split_k)}


def merge_splitk_override_tables(
    tables: list[dict[str, int]] | tuple[dict[str, int], ...],
) -> dict[str, int]:
    """Merge strict split-k override tables, rejecting contradictory entries."""
    merged: dict[str, int] = {}
    for table in tables:
        for key, value in table.items():
            split_k = int(value)
            if split_k < 1:
                raise ValueError("split-k override values must be >= 1")
            if key in merged and merged[key] != split_k:
                raise ValueError(f"Conflicting split-k override for {key!r}")
            merged[key] = split_k
    return merged


def summarize_runtime_gate(
    *,
    blockwise_summary: dict[str, Any],
    throughput_v2_summary: dict[str, Any],
) -> dict[str, Any]:
    blockwise_timings = blockwise_summary.get("timings", {})
    throughput_timings = throughput_v2_summary.get("timings", {})
    blockwise_outputs = blockwise_summary.get("outputs", {})
    throughput_outputs = throughput_v2_summary.get("outputs", {})
    outputs_match = (
        blockwise_outputs.get("records") == throughput_outputs.get("records")
    )
    return {
        "format_version": 1,
        "blockwise_summary_path": blockwise_summary.get("summary_path"),
        "throughput_v2_summary_path": throughput_v2_summary.get("summary_path"),
        "outputs_match": outputs_match,
        "blockwise_output_records": blockwise_outputs.get("records"),
        "throughput_v2_output_records": throughput_outputs.get("records"),
        "timings": {
            "top_level_total_s": {
                "blockwise": blockwise_timings.get("top_level_total_s"),
                "throughput_v2": throughput_timings.get("top_level_total_s"),
                "delta": (
                    throughput_timings.get("top_level_total_s")
                    - blockwise_timings.get("top_level_total_s")
                    if (
                        blockwise_timings.get("top_level_total_s") is not None
                        and throughput_timings.get("top_level_total_s") is not None
                    )
                    else None
                ),
            },
            "model_execute_total_s": {
                "blockwise": blockwise_timings.get("model_execute_total_s"),
                "throughput_v2": throughput_timings.get("model_execute_total_s"),
                "delta": (
                    throughput_timings.get("model_execute_total_s")
                    - blockwise_timings.get("model_execute_total_s")
                    if (
                        blockwise_timings.get("model_execute_total_s") is not None
                        and throughput_timings.get("model_execute_total_s") is not None
                    )
                    else None
                ),
            },
        },
    }
