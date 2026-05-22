from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from diagnose_throughput_v2_token_divergence import (
    VARIANT_ORDER,
    build_workload_prompts,
    find_first_token_divergence,
    summarize_divergent_logits,
)
from nanovllm_jax.engine.model_runner import ModelRunner


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_build_workload_prompts_repeats_base_prompts_for_canary() -> None:
    prompts = build_workload_prompts(
        workload="canary",
        base_prompts=["a", "b", "c"],
        canary_batch_size=7,
    )

    assert prompts == ["a", "b", "c", "a", "b", "c", "a"]


def test_find_first_token_divergence_is_step_major() -> None:
    divergence = find_first_token_divergence(
        {
            "blockwise": [[1, 2, 3], [4, 5, 6]],
            "throughput_v2_jax": [[1, 2, 3], [4, 9, 6]],
            "throughput_v2_mosaic": [[1, 2, 8], [4, 5, 6]],
        },
    )

    assert divergence == {
        "prompt_index": 1,
        "token_index": 1,
        "generation_step_index": 1,
        "tokens": {
            "blockwise": 5,
            "throughput_v2_jax": 9,
            "throughput_v2_mosaic": 5,
        },
        "reference_variant": "blockwise",
        "reference_token_id": 5,
    }


def test_model_runner_generation_debug_reports_topk_and_saves_logits(tmp_path: Path) -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner._last_generation_debug = {
        "phase": "decode",
        "is_prefill": False,
        "real_batch_size": 2,
        "padded_batch_size": 2,
        "seq_ids": [0, 1],
        "prefix_token_ids": [[10, 11], [20, 21]],
        "logits": jnp.asarray(
            [
                [0.0, 3.0, 1.0, 2.0],
                [4.0, 0.5, 5.0, -1.0],
            ],
            dtype=jnp.float32,
        ),
        "sampled_token_ids": np.asarray([3, 2], dtype=np.int64),
    }

    logits_path = tmp_path / "logits.npy"
    payload = runner.consume_last_generation_debug(
        top_k=3,
        save_logits_path=logits_path,
    )

    assert payload is not None
    assert payload["argmax_token_ids"] == [1, 2]
    assert payload["argmax_margins"] == [1.0, 1.0]
    assert payload["sampled_token_ranks"] == [2, 1]
    assert payload["top_token_ids"] == [[1, 3, 2], [2, 0, 1]]
    assert logits_path.exists()
    assert np.load(logits_path).shape == (2, 4)
    assert runner.consume_last_generation_debug() is None


def test_summarize_divergent_logits_computes_max_diff_and_row_details(
    tmp_path: Path,
) -> None:
    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return " ".join(str(token) for token in token_ids)

    step_index = 2
    prompt_index = 1
    records = {
        "blockwise": {
            "generation_step_index": step_index,
            "sampled_token_ids": [7, 8],
            "argmax_token_ids": [7, 8],
            "argmax_logits": [10.0, 9.0],
            "argmax_margins": [1.0, 0.5],
            "sampled_token_logits": [10.0, 9.0],
            "sampled_token_ranks": [1, 1],
            "top_token_ids": [[7, 3], [8, 2]],
            "top_logits": [[10.0, 9.0], [9.0, 8.5]],
            "prefix_token_ids": [[1, 2], [3, 4]],
        },
        "throughput_v2_jax": {
            "generation_step_index": step_index,
            "sampled_token_ids": [7, 9],
            "argmax_token_ids": [7, 9],
            "argmax_logits": [10.0, 9.25],
            "argmax_margins": [1.0, 0.25],
            "sampled_token_logits": [10.0, 9.25],
            "sampled_token_ranks": [1, 1],
            "top_token_ids": [[7, 3], [9, 8]],
            "top_logits": [[10.0, 9.0], [9.25, 9.0]],
            "prefix_token_ids": [[1, 2], [3, 4]],
        },
        "throughput_v2_mosaic": {
            "generation_step_index": step_index,
            "sampled_token_ids": [7, 8],
            "argmax_token_ids": [7, 8],
            "argmax_logits": [10.1, 8.9],
            "argmax_margins": [1.0, 0.5],
            "sampled_token_logits": [10.1, 8.9],
            "sampled_token_ranks": [1, 1],
            "top_token_ids": [[7, 3], [8, 2]],
            "top_logits": [[10.1, 9.1], [8.9, 8.4]],
            "prefix_token_ids": [[1, 2], [3, 4]],
        },
    }
    logits = {
        "blockwise": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "throughput_v2_jax": np.asarray([[1.5, 2.0], [3.0, 5.0]], dtype=np.float32),
        "throughput_v2_mosaic": np.asarray([[1.0, 2.25], [2.5, 4.0]], dtype=np.float32),
    }
    for variant in VARIANT_ORDER:
        variant_dir = tmp_path / f"{variant}_logits_step_{step_index}"
        _write_jsonl(variant_dir / "generation_debug.jsonl", [records[variant]])
        np.save(variant_dir / f"logits_step_{step_index}.npy", logits[variant])

    summary = summarize_divergent_logits(
        workload_dir=tmp_path,
        divergence={
            "prompt_index": prompt_index,
            "token_index": step_index,
            "generation_step_index": step_index,
            "tokens": {
                "blockwise": 8,
                "throughput_v2_jax": 9,
                "throughput_v2_mosaic": 8,
            },
        },
        tokenizer=Tokenizer(),
        keep_full_logits=False,
    )

    assert summary["max_logit_diffs_vs_blockwise"]["throughput_v2_jax"] == {
        "all_rows_max_abs_diff": 1.0,
        "divergent_row_max_abs_diff": 1.0,
    }
    assert summary["max_logit_diffs_vs_blockwise"]["throughput_v2_mosaic"] == {
        "all_rows_max_abs_diff": 0.5,
        "divergent_row_max_abs_diff": 0.5,
    }
    assert summary["variant_details"]["throughput_v2_jax"]["sampled_token_id"] == 9
    assert summary["variant_details"]["blockwise"]["prefix_text"] == "3 4"
    assert not (tmp_path / f"blockwise_logits_step_{step_index}" / f"logits_step_{step_index}.npy").exists()
