"""Tests for the shipped decode-profile artifact helper."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanovllm_jax.utils.decode_profile_artifacts import (
    compare_decode_profile_summaries,
    run_controlled_decode_profile,
    run_kv_update_backend_matrix,
    summarize_decode_profile_runs,
)


class DummySamplingParams:
    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens


class DummyLLM:
    instances = []

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.exited = False
        DummyLLM.instances.append(self)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert use_tqdm is False
        raw_dir = Path(os.environ["NANOVLLM_JAX_DIAGNOSTICS_DIR"])
        decode_step_path = raw_dir / "decode_step_profile.jsonl"
        decode_schedule_path = raw_dir / "decode_schedule.jsonl"
        decode_step_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "decode_step_profile",
                            "decode_input_action": "patch_active_rows",
                            "prepared_metadata_action": "retain",
                            "block_table_action": "reuse_block_tables",
                            "kv_update_calls": 1,
                            "kv_update_backend": "scatter",
                            "kv_update_tokens": 2,
                            "kv_update_valid_tokens": 1,
                            "kv_update_skipped_tokens": 1,
                            "kv_update_duplicate_slots": 0,
                            "kv_update_measured": True,
                            "partitioned_decode_reduction_backend": "streaming",
                            "partitioned_decode_reduction_family": "throughput",
                            "partitioned_decode_reduction_measured": True,
                            "scheduler_s": 0.001,
                            "prepare_decode_s": 0.002,
                            "model_execute_s": 0.010,
                            "kv_update_s": 0.003,
                            "partitioned_decode_reduction_s": 0.0015,
                            "sampler_s": 0.004,
                            "postprocess_s": 0.0005,
                        }
                    ),
                    json.dumps(
                        {
                            "event": "decode_step_profile",
                            "decode_input_action": "patch_active_rows",
                            "prepared_metadata_action": "clear_schedule_changed",
                            "block_table_action": "patch_block_table_rows",
                            "kv_update_calls": 1,
                            "kv_update_backend": "scatter",
                            "kv_update_tokens": 2,
                            "kv_update_valid_tokens": 2,
                            "kv_update_skipped_tokens": 0,
                            "kv_update_duplicate_slots": 1,
                            "kv_update_measured": True,
                            "partitioned_decode_reduction_backend": "streaming",
                            "partitioned_decode_reduction_family": "throughput",
                            "partitioned_decode_reduction_measured": True,
                            "scheduler_s": 0.0015,
                            "prepare_decode_s": 0.0025,
                            "model_execute_s": 0.012,
                            "kv_update_s": 0.004,
                            "partitioned_decode_reduction_s": 0.002,
                            "sampler_s": 0.0045,
                            "postprocess_s": 0.0007,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        decode_schedule_path.write_text(
            json.dumps(
                {
                    "event": "decode_schedule_refresh",
                    "action": "reuse_block_tables",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert sampling_params.temperature == 0.0
        assert sampling_params.max_tokens == 12
        assert os.environ.get("NANOVLLM_JAX_KV_UPDATE_BACKEND") == "sorted_compact_scatter"
        assert os.environ.get("NANOVLLM_JAX_MOSAIC_DECODE_KERNEL") == "throughput"
        assert os.environ.get("NANOVLLM_JAX_INTERNAL_MOSAIC_MIN_DECODE_BATCH") == "0"
        assert (
            os.environ.get("NANOVLLM_JAX_INTERNAL_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH")
            == "0"
        )
        return [{"text": "", "token_ids": [1, 2, 3]} for _ in prompts]

    def exit(self):
        self.exited = True


class DummyMatrixLLM:
    instances = []

    _PROFILES = {
        "scatter": {
            "kv_update_s": 0.004,
            "kv_update_calls": 1,
            "kv_update_tokens": 4,
            "kv_update_valid_tokens": 3,
            "kv_update_skipped_tokens": 1,
            "kv_update_duplicate_slots": 1,
        },
        "sorted_compact_scatter": {
            "kv_update_s": 0.003,
            "kv_update_calls": 1,
            "kv_update_tokens": 4,
            "kv_update_valid_tokens": 3,
            "kv_update_skipped_tokens": 1,
            "kv_update_duplicate_slots": 0,
        },
    }

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.exited = False
        DummyMatrixLLM.instances.append(self)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert use_tqdm is False
        backend = os.environ["NANOVLLM_JAX_KV_UPDATE_BACKEND"]
        profile = self._PROFILES[backend]
        raw_dir = Path(os.environ["NANOVLLM_JAX_DIAGNOSTICS_DIR"])
        (raw_dir / "decode_step_profile.jsonl").write_text(
            json.dumps(
                {
                    "event": "decode_step_profile",
                    "decode_input_action": "patch_active_rows",
                    "block_table_action": "reuse_block_tables",
                    "kv_update_backend": backend,
                    "kv_update_calls": profile["kv_update_calls"],
                    "kv_update_tokens": profile["kv_update_tokens"],
                    "kv_update_valid_tokens": profile["kv_update_valid_tokens"],
                    "kv_update_skipped_tokens": profile["kv_update_skipped_tokens"],
                    "kv_update_duplicate_slots": profile["kv_update_duplicate_slots"],
                    "kv_update_measured": True,
                    "scheduler_s": 0.001,
                    "prepare_decode_s": 0.002,
                    "model_execute_s": 0.010,
                    "kv_update_s": profile["kv_update_s"],
                    "sampler_s": 0.003,
                    "postprocess_s": 0.0005,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (raw_dir / "decode_schedule.jsonl").write_text(
            json.dumps(
                {
                    "event": "decode_schedule_refresh",
                    "action": "reuse_block_tables",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert sampling_params.max_tokens == 8
        return [{"text": "", "token_ids": [1, 2]} for _ in prompts]

    def exit(self):
        self.exited = True


class DummyNoArtifactsLLM:
    instances = []

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.exited = False
        DummyNoArtifactsLLM.instances.append(self)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert use_tqdm is False
        assert sampling_params.temperature == 0.0
        assert sampling_params.max_tokens == 4
        return [{"text": "", "token_ids": [1]} for _ in prompts]

    def exit(self):
        self.exited = True


class DummySparseArtifactsLLM:
    instances = []

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.exited = False
        DummySparseArtifactsLLM.instances.append(self)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert use_tqdm is False
        raw_dir = Path(os.environ["NANOVLLM_JAX_DIAGNOSTICS_DIR"])
        (raw_dir / "decode_step_profile.jsonl").write_text(
            "\n"
            + json.dumps(
                {
                    "event": "decode_step_profile",
                    "decode_input_action": "patch_active_rows",
                    "prepare_decode_s": 0.002,
                }
            )
            + "\n\n",
            encoding="utf-8",
        )
        (raw_dir / "decode_schedule.jsonl").write_text("\n\n", encoding="utf-8")
        assert sampling_params.temperature == 0.0
        assert sampling_params.max_tokens == 6
        return [{"text": "", "token_ids": [1, 2]} for _ in prompts]

    def exit(self):
        self.exited = True


def test_run_controlled_decode_profile_writes_stable_summary(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "artifacts"

    monkeypatch.delenv("NANOVLLM_JAX_PROFILE_DECODE_STEP", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DUMP_DECODE_SCHEDULE", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DIAGNOSTICS_DIR", raising=False)
    DummyLLM.instances.clear()

    summary = run_controlled_decode_profile(
        model_path=model_dir,
        output_dir=output_dir,
        prompts=["a", "b"],
        max_tokens=12,
        temperature=0.0,
        decode_attention_backend="blockwise",
        kv_update_backend="sorted_compact_scatter",
        mosaic_kernel_family="throughput",
        mosaic_min_decode_batch=0,
        mosaic_throughput_min_decode_batch=0,
        enforce_eager=True,
        invocation={
            "argv": ["profile_decode_runtime.py", "--decode-backend", "blockwise"],
            "cwd": str(tmp_path),
            "entrypoint": "profile_decode_runtime.py",
        },
        llm_class=DummyLLM,
        sampling_params_cls=DummySamplingParams,
    )

    summary_path = output_dir / "decode_runtime_profile_summary.json"
    manifest_path = output_dir / "run_manifest.json"
    assert summary_path.is_file()
    assert manifest_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["format_version"] == 1
    assert payload["runtime"]["decode_attention_backend"] == "blockwise"
    assert payload["runtime"]["kv_update_backend"] == "sorted_compact_scatter"
    assert payload["runtime"]["mosaic_kernel_family"] == "throughput"
    assert payload["runtime"]["mosaic_min_decode_batch"] == 0
    assert payload["runtime"]["mosaic_throughput_min_decode_batch"] == 0
    assert payload["runtime"]["enforce_eager"] is True
    assert payload["runtime"]["tensor_parallel_size"] == 1
    assert payload["prompts"]["count"] == 2
    assert payload["outputs"]["count"] == 2
    assert payload["outputs"]["token_counts"] == [3, 3]
    assert payload["artifacts"]["decode_step_records"] == 2
    assert payload["artifacts"]["decode_schedule_records"] == 1
    assert payload["artifacts"]["run_manifest_path"] == str(manifest_path)
    assert payload["histograms"]["decode_input_actions"] == {"patch_active_rows": 2}
    assert payload["histograms"]["prepared_metadata_actions"] == {
        "clear_schedule_changed": 1,
        "retain": 1,
    }
    assert payload["histograms"]["block_table_actions"] == {
        "patch_block_table_rows": 1,
        "reuse_block_tables": 1,
    }
    assert payload["histograms"]["kv_update_measured"] == {"True": 2}
    assert payload["histograms"]["decode_schedule_actions"] == {
        "reuse_block_tables": 1,
    }
    assert payload["histograms"]["partitioned_decode_reduction_backends"] == {
        "streaming": 2,
    }
    assert payload["histograms"]["partitioned_decode_reduction_families"] == {
        "throughput": 2,
    }
    assert payload["histograms"]["partitioned_decode_reduction_measured"] == {
        "True": 2,
    }
    assert payload["counters"]["kv_update_calls"] == 2
    assert payload["counters"]["kv_update_tokens"] == 4
    assert payload["counters"]["kv_update_valid_tokens"] == 3
    assert payload["counters"]["kv_update_skipped_tokens"] == 1
    assert payload["counters"]["kv_update_duplicate_slots"] == 1
    assert payload["counters"]["kv_update_valid_token_pct"] == pytest.approx(75.0)
    assert payload["counters"]["kv_update_skipped_token_pct"] == pytest.approx(25.0)
    assert payload["counters"]["kv_update_duplicate_slot_pct"] == pytest.approx(
        33.33333333333333
    )
    assert payload["timings"]["decode_step_count"] == 2
    assert payload["timings"]["top_level"]["prepare_decode_s"]["records_with_value"] == 2
    assert payload["timings"]["top_level"]["prepare_decode_s"]["total_s"] == pytest.approx(0.0045)
    assert payload["timings"]["model_execute_subcomponents"]["kv_update_s"]["total_s"] == pytest.approx(0.007)
    assert payload["timings"]["model_execute_subcomponents"]["partitioned_decode_reduction_s"]["share_of_model_execute_pct"] is not None
    assert payload["manifest"]["runtime"]["kv_update_backend"] == "sorted_compact_scatter"
    assert payload["manifest"]["runtime"]["mosaic_kernel_family"] == "throughput"
    assert payload["manifest"]["invocation"]["entrypoint"] == "profile_decode_runtime.py"
    assert payload["manifest"]["env"]["NANOVLLM_JAX_KV_UPDATE_BACKEND"] == "sorted_compact_scatter"
    assert payload["manifest"]["env"]["NANOVLLM_JAX_MOSAIC_DECODE_KERNEL"] == "throughput"
    assert manifest_payload["runtime"]["decode_attention_backend"] == "blockwise"
    assert manifest_payload["invocation"]["cwd"] == str(tmp_path)

    assert summary["summary_path"] == str(summary_path)
    assert DummyLLM.instances
    assert DummyLLM.instances[-1].kwargs["max_num_seqs"] == 2
    assert DummyLLM.instances[-1].kwargs["tensor_parallel_size"] == 1
    assert DummyLLM.instances[-1].exited is True
    assert "NANOVLLM_JAX_DIAGNOSTICS_DIR" not in os.environ


def test_run_controlled_decode_profile_handles_missing_artifacts_and_default_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "artifacts"

    monkeypatch.delenv("NANOVLLM_JAX_PROFILE_DECODE_STEP", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DUMP_DECODE_SCHEDULE", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DIAGNOSTICS_DIR", raising=False)
    DummyNoArtifactsLLM.instances.clear()

    summary = run_controlled_decode_profile(
        model_path=model_dir,
        output_dir=output_dir,
        prompts=["a"],
        max_tokens=4,
        temperature=0.0,
        llm_class=DummyNoArtifactsLLM,
        sampling_params_cls=DummySamplingParams,
    )

    assert summary["runtime"]["decode_attention_backend"] == "auto"
    assert summary["runtime"]["kv_update_backend"] == "default"
    assert summary["runtime"]["mosaic_kernel_family"] == "default"
    assert summary["runtime"]["mosaic_min_decode_batch"] == "default"
    assert summary["runtime"]["mosaic_throughput_min_decode_batch"] == "default"
    assert summary["runtime"]["enforce_eager"] is False
    assert summary["artifacts"]["decode_step_records"] == 0
    assert summary["artifacts"]["decode_schedule_records"] == 0
    assert summary["histograms"]["decode_input_actions"] == {}
    assert summary["histograms"]["decode_schedule_actions"] == {}
    assert summary["histograms"]["kv_update_backends"] == {}
    assert summary["counters"]["kv_update_calls"] == 0
    assert summary["counters"]["kv_update_tokens"] == 0
    assert summary["counters"]["kv_update_valid_token_pct"] is None
    assert summary["timings"]["decode_step_count"] == 0
    assert summary["timings"]["top_level_total_s"] == 0.0
    assert summary["timings"]["model_execute_total_s"] == 0.0
    assert DummyNoArtifactsLLM.instances[-1].exited is True


def test_run_controlled_decode_profile_handles_empty_and_sparse_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "artifacts"

    monkeypatch.delenv("NANOVLLM_JAX_PROFILE_DECODE_STEP", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DUMP_DECODE_SCHEDULE", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DIAGNOSTICS_DIR", raising=False)
    DummySparseArtifactsLLM.instances.clear()

    summary = run_controlled_decode_profile(
        model_path=model_dir,
        output_dir=output_dir,
        prompts=["a"],
        max_tokens=6,
        temperature=0.0,
        llm_class=DummySparseArtifactsLLM,
        sampling_params_cls=DummySamplingParams,
    )

    assert summary["artifacts"]["decode_step_records"] == 1
    assert summary["artifacts"]["decode_schedule_records"] == 0
    assert summary["histograms"]["decode_input_actions"] == {"patch_active_rows": 1}
    assert summary["histograms"]["decode_schedule_actions"] == {}
    assert summary["histograms"]["kv_update_backends"] == {}
    assert summary["counters"]["kv_update_calls"] == 0
    assert summary["counters"]["kv_update_tokens"] == 0
    assert summary["counters"]["kv_update_valid_tokens"] == 0
    assert summary["counters"]["kv_update_duplicate_slot_pct"] is None
    assert summary["timings"]["decode_step_count"] == 1
    assert summary["timings"]["top_level"]["prepare_decode_s"]["records_with_value"] == 1
    assert summary["timings"]["top_level"]["prepare_decode_s"]["total_s"] == pytest.approx(
        0.002
    )
    assert DummySparseArtifactsLLM.instances[-1].exited is True


def test_run_kv_update_backend_matrix_writes_comparisons(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "matrix"

    monkeypatch.delenv("NANOVLLM_JAX_PROFILE_DECODE_STEP", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DUMP_DECODE_SCHEDULE", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DIAGNOSTICS_DIR", raising=False)
    DummyMatrixLLM.instances.clear()

    matrix = run_kv_update_backend_matrix(
        model_path=model_dir,
        output_dir=output_dir,
        backends=["scatter", "sorted_compact_scatter"],
        baseline_backend="scatter",
        prompts=["hello"],
        max_tokens=8,
        temperature=0.0,
        decode_attention_backend="blockwise",
        enforce_eager=True,
        invocation={
            "argv": ["profile_kv_update_backends.py"],
            "cwd": str(tmp_path),
            "entrypoint": "profile_kv_update_backends.py",
        },
        llm_class=DummyMatrixLLM,
        sampling_params_cls=DummySamplingParams,
    )

    matrix_path = output_dir / "kv_update_backend_matrix.json"
    comparison_path = output_dir / "compare_scatter_vs_sorted_compact_scatter.json"
    assert matrix_path.is_file()
    assert comparison_path.is_file()

    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert payload["baseline_backend"] == "scatter"
    assert payload["backend_order"] == ["scatter", "sorted_compact_scatter"]
    assert payload["runtime"]["decode_attention_backend"] == "blockwise"
    assert payload["comparisons_vs_baseline"]["sorted_compact_scatter"][
        "kv_update_total_s_delta"
    ] == pytest.approx(-0.001)
    assert payload["comparisons_vs_baseline"]["sorted_compact_scatter"][
        "kv_update_valid_tokens_delta"
    ] == 0.0
    assert payload["comparisons_vs_baseline"]["sorted_compact_scatter"][
        "kv_update_duplicate_slots_delta"
    ] == -1.0
    assert payload["aggregate"]["counters"]["kv_update_calls"] == 2
    assert payload["aggregate"]["counters"]["kv_update_valid_tokens"] == 6
    assert payload["aggregate"]["counters"]["kv_update_duplicate_slots"] == 1
    assert payload["runs"][0]["backend"] == "scatter"
    assert payload["runs"][1]["backend"] == "sorted_compact_scatter"
    assert payload["runs"][1]["counters"]["kv_update_duplicate_slots"] == 0

    assert matrix["matrix_path"] == str(matrix_path)
    assert DummyMatrixLLM.instances[-1].exited is True


def test_run_kv_update_backend_matrix_validates_backend_arguments(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "matrix"

    with pytest.raises(ValueError, match="at least one backend"):
        run_kv_update_backend_matrix(
            model_path=model_dir,
            output_dir=output_dir,
            backends=[],
            llm_class=DummyMatrixLLM,
            sampling_params_cls=DummySamplingParams,
        )

    with pytest.raises(ValueError, match="must be included in backends"):
        run_kv_update_backend_matrix(
            model_path=model_dir,
            output_dir=output_dir,
            backends=["scatter", "sorted_compact_scatter"],
            baseline_backend="compact_scatter",
            llm_class=DummyMatrixLLM,
            sampling_params_cls=DummySamplingParams,
        )


def test_run_kv_update_backend_matrix_preserves_custom_backend_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "matrix"

    monkeypatch.delenv("NANOVLLM_JAX_PROFILE_DECODE_STEP", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DUMP_DECODE_SCHEDULE", raising=False)
    monkeypatch.delenv("NANOVLLM_JAX_DIAGNOSTICS_DIR", raising=False)
    DummyMatrixLLM.instances.clear()

    matrix = run_kv_update_backend_matrix(
        model_path=model_dir,
        output_dir=output_dir,
        backends=["sorted_compact_scatter", "scatter"],
        baseline_backend="sorted_compact_scatter",
        prompts=["hello"],
        max_tokens=8,
        temperature=0.0,
        decode_attention_backend="blockwise",
        enforce_eager=True,
        invocation={
            "argv": ["profile_kv_update_backends.py"],
            "cwd": str(tmp_path),
            "entrypoint": "profile_kv_update_backends.py",
        },
        llm_class=DummyMatrixLLM,
        sampling_params_cls=DummySamplingParams,
    )

    assert matrix["backend_order"] == ["sorted_compact_scatter", "scatter"]
    assert [run["backend"] for run in matrix["runs"]] == [
        "sorted_compact_scatter",
        "scatter",
    ]
    assert set(matrix["comparisons_vs_baseline"]) == {"scatter"}
    assert (
        output_dir / "compare_sorted_compact_scatter_vs_scatter.json"
    ).is_file()
    assert not (
        output_dir / "compare_sorted_compact_scatter_vs_sorted_compact_scatter.json"
    ).exists()


def test_compare_decode_profile_summaries_returns_stable_histogram_diff(tmp_path: Path) -> None:
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"

    before_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "decode_step_records": 4,
                    "decode_schedule_records": 2,
                },
                "histograms": {
                    "decode_input_actions": {
                        "full_transfer": 1,
                        "patch_active_rows": 3,
                    },
                    "prepared_metadata_actions": {
                        "clear_initial": 1,
                        "retain": 3,
                    },
                    "block_table_actions": {
                        "full_transfer": 2,
                        "reuse_block_tables": 2,
                    },
                    "kv_update_backends": {
                        "scatter": 4,
                    },
                    "kv_update_measured": {
                        "True": 4,
                    },
                    "partitioned_decode_reduction_backends": {
                        "streaming": 1,
                    },
                    "partitioned_decode_reduction_families": {
                        "latency": 1,
                    },
                    "partitioned_decode_reduction_measured": {
                        "False": 4,
                    },
                    "decode_schedule_actions": {
                        "rebuild": 1,
                        "reuse_block_tables": 1,
                    },
                },
                "runtime": {
                    "decode_attention_backend": "auto",
                },
                "counters": {
                    "kv_update_calls": 4,
                    "kv_update_tokens": 16,
                    "kv_update_valid_tokens": 12,
                    "kv_update_skipped_tokens": 4,
                    "kv_update_duplicate_slots": 2,
                    "kv_update_valid_token_pct": 75.0,
                    "kv_update_skipped_token_pct": 25.0,
                    "kv_update_duplicate_slot_pct": 16.666666666666664,
                },
                "timings": {
                    "decode_step_count": 4,
                    "top_level_total_s": 0.40,
                    "model_execute_total_s": 0.20,
                    "top_level": {
                        "prepare_decode_s": {
                            "records_with_value": 4,
                            "total_s": 0.08,
                            "mean_s": 0.02,
                            "max_s": 0.03,
                            "share_of_top_level_pct": 20.0,
                            "share_of_model_execute_pct": None,
                        },
                    },
                    "model_execute_subcomponents": {
                        "kv_update_s": {
                            "records_with_value": 4,
                            "total_s": 0.04,
                            "mean_s": 0.01,
                            "max_s": 0.015,
                            "share_of_top_level_pct": 10.0,
                            "share_of_model_execute_pct": 20.0,
                        },
                    },
                },
                "prompts": {
                    "count": 2,
                },
                "outputs": {
                    "count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "decode_step_records": 5,
                    "decode_schedule_records": 3,
                },
                "histograms": {
                    "decode_input_actions": {
                        "patch_active_rows": 5,
                    },
                    "prepared_metadata_actions": {
                        "clear_schedule_changed": 2,
                        "retain": 3,
                    },
                    "block_table_actions": {
                        "patch_block_table_rows": 3,
                        "reuse_block_tables": 2,
                    },
                    "kv_update_backends": {
                        "compact_scatter": 5,
                    },
                    "kv_update_measured": {
                        "False": 5,
                    },
                    "partitioned_decode_reduction_backends": {
                        "streaming": 2,
                    },
                    "partitioned_decode_reduction_families": {
                        "throughput": 2,
                    },
                    "partitioned_decode_reduction_measured": {
                        "True": 5,
                    },
                    "decode_schedule_actions": {
                        "reuse_block_tables": 3,
                    },
                },
                "runtime": {
                    "decode_attention_backend": "mosaic",
                },
                "counters": {
                    "kv_update_calls": 5,
                    "kv_update_tokens": 20,
                    "kv_update_valid_tokens": 18,
                    "kv_update_skipped_tokens": 2,
                    "kv_update_duplicate_slots": 3,
                    "kv_update_valid_token_pct": 90.0,
                    "kv_update_skipped_token_pct": 10.0,
                    "kv_update_duplicate_slot_pct": 16.666666666666664,
                },
                "timings": {
                    "decode_step_count": 5,
                    "top_level_total_s": 0.50,
                    "model_execute_total_s": 0.30,
                    "top_level": {
                        "prepare_decode_s": {
                            "records_with_value": 5,
                            "total_s": 0.05,
                            "mean_s": 0.01,
                            "max_s": 0.012,
                            "share_of_top_level_pct": 10.0,
                            "share_of_model_execute_pct": None,
                        },
                    },
                    "model_execute_subcomponents": {
                        "kv_update_s": {
                            "records_with_value": 5,
                            "total_s": 0.09,
                            "mean_s": 0.018,
                            "max_s": 0.02,
                            "share_of_top_level_pct": 18.0,
                            "share_of_model_execute_pct": 30.0,
                        },
                    },
                },
                "prompts": {
                    "count": 2,
                },
                "outputs": {
                    "count": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_decode_profile_summaries(before_path, after_path)

    assert comparison["format_version"] == 1
    assert comparison["record_counts"]["decode_step_records"] == {
        "before": 4,
        "after": 5,
        "delta": 1,
    }
    assert comparison["record_counts"]["decode_schedule_records"] == {
        "before": 2,
        "after": 3,
        "delta": 1,
    }
    assert comparison["histograms"]["decode_input_actions"] == {
        "full_transfer": {"before": 1, "after": 0, "delta": -1},
        "patch_active_rows": {"before": 3, "after": 5, "delta": 2},
    }
    assert comparison["histograms"]["prepared_metadata_actions"] == {
        "clear_initial": {"before": 1, "after": 0, "delta": -1},
        "clear_schedule_changed": {"before": 0, "after": 2, "delta": 2},
        "retain": {"before": 3, "after": 3, "delta": 0},
    }
    assert comparison["histograms"]["block_table_actions"] == {
        "full_transfer": {"before": 2, "after": 0, "delta": -2},
        "patch_block_table_rows": {"before": 0, "after": 3, "delta": 3},
        "reuse_block_tables": {"before": 2, "after": 2, "delta": 0},
    }
    assert comparison["histograms"]["kv_update_backends"] == {
        "compact_scatter": {"before": 0, "after": 5, "delta": 5},
        "scatter": {"before": 4, "after": 0, "delta": -4},
    }
    assert comparison["histograms"]["kv_update_measured"] == {
        "False": {"before": 0, "after": 5, "delta": 5},
        "True": {"before": 4, "after": 0, "delta": -4},
    }
    assert comparison["histograms"]["partitioned_decode_reduction_backends"] == {
        "streaming": {"before": 1, "after": 2, "delta": 1},
    }
    assert comparison["histograms"]["partitioned_decode_reduction_families"] == {
        "latency": {"before": 1, "after": 0, "delta": -1},
        "throughput": {"before": 0, "after": 2, "delta": 2},
    }
    assert comparison["histograms"]["partitioned_decode_reduction_measured"] == {
        "False": {"before": 4, "after": 0, "delta": -4},
        "True": {"before": 0, "after": 5, "delta": 5},
    }
    assert comparison["histograms"]["decode_schedule_actions"] == {
        "rebuild": {"before": 1, "after": 0, "delta": -1},
        "reuse_block_tables": {"before": 1, "after": 3, "delta": 2},
    }
    assert comparison["context"]["before_runtime"] == {
        "decode_attention_backend": "auto",
    }
    assert comparison["context"]["after_runtime"] == {
        "decode_attention_backend": "mosaic",
    }
    assert comparison["counters"]["kv_update_calls"] == {
        "before": 4.0,
        "after": 5.0,
        "delta": 1.0,
    }
    assert comparison["counters"]["kv_update_valid_tokens"] == {
        "before": 12.0,
        "after": 18.0,
        "delta": 6.0,
    }
    assert comparison["counters"]["kv_update_skipped_token_pct"] == {
        "before": 25.0,
        "after": 10.0,
        "delta": -15.0,
    }
    assert comparison["timings"]["decode_step_count"] == {
        "before": 4.0,
        "after": 5.0,
        "delta": 1.0,
    }
    prepare_total = comparison["timings"]["top_level"]["prepare_decode_s"]["total_s"]
    assert prepare_total["before"] == 0.08
    assert prepare_total["after"] == 0.05
    assert prepare_total["delta"] == pytest.approx(-0.03)
    kv_share = comparison["timings"]["model_execute_subcomponents"]["kv_update_s"][
        "share_of_model_execute_pct"
    ]
    assert kv_share["before"] == 20.0
    assert kv_share["after"] == 30.0
    assert kv_share["delta"] == 10.0


def test_compare_and_summarize_tolerate_missing_counter_sections(tmp_path: Path) -> None:
    before_path = tmp_path / "before_sparse.json"
    after_path = tmp_path / "after_sparse.json"

    before_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "decode_step_records": 1,
                    "decode_schedule_records": 0,
                },
                "histograms": {
                    "decode_input_actions": {"patch_active_rows": 1},
                },
                "runtime": {
                    "decode_attention_backend": "auto",
                },
                "timings": {
                    "decode_step_count": 1,
                    "top_level_total_s": 0.1,
                    "model_execute_total_s": 0.05,
                    "top_level": {},
                    "model_execute_subcomponents": {},
                },
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "decode_step_records": 2,
                    "decode_schedule_records": 0,
                },
                "histograms": {
                    "decode_input_actions": {"patch_active_rows": 2},
                },
                "runtime": {
                    "decode_attention_backend": "blockwise",
                },
                "timings": {
                    "decode_step_count": 2,
                    "top_level_total_s": 0.2,
                    "model_execute_total_s": 0.07,
                    "top_level": {},
                    "model_execute_subcomponents": {},
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_decode_profile_summaries(before_path, after_path)
    aggregate = summarize_decode_profile_runs([before_path, after_path])

    assert comparison["counters"]["kv_update_calls"] == {
        "before": None,
        "after": None,
        "delta": None,
    }
    assert comparison["counters"]["kv_update_valid_tokens"] == {
        "before": None,
        "after": None,
        "delta": None,
    }
    assert aggregate["counters"]["kv_update_calls"] == 0
    assert aggregate["counters"]["kv_update_tokens"] == 0
    assert aggregate["counters"]["kv_update_valid_tokens"] == 0
    assert aggregate["counters"]["kv_update_valid_token_pct"] is None


def test_summarize_decode_profile_runs_aggregates_timing_breakdown(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a.json"
    run_b = tmp_path / "run_b.json"
    run_a.write_text(
        json.dumps(
            {
                "summary_path": str(run_a),
                "runtime": {
                    "decode_attention_backend": "auto",
                    "kv_update_backend": "scatter",
                },
                "counters": {
                    "kv_update_calls": 3,
                    "kv_update_tokens": 12,
                    "kv_update_valid_tokens": 9,
                    "kv_update_skipped_tokens": 3,
                    "kv_update_duplicate_slots": 1,
                    "kv_update_valid_token_pct": 75.0,
                    "kv_update_skipped_token_pct": 25.0,
                    "kv_update_duplicate_slot_pct": 11.11111111111111,
                },
                "timings": {
                    "decode_step_count": 3,
                    "top_level_total_s": 0.30,
                    "model_execute_total_s": 0.18,
                    "top_level": {
                        "prepare_decode_s": {
                            "records_with_value": 3,
                            "total_s": 0.06,
                            "mean_s": 0.02,
                            "max_s": 0.025,
                        },
                    },
                    "model_execute_subcomponents": {
                        "kv_update_s": {
                            "records_with_value": 3,
                            "total_s": 0.03,
                            "mean_s": 0.01,
                            "max_s": 0.012,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    run_b.write_text(
        json.dumps(
            {
                "summary_path": str(run_b),
                "runtime": {
                    "decode_attention_backend": "mosaic",
                    "kv_update_backend": "sorted_compact_scatter",
                },
                "counters": {
                    "kv_update_calls": 5,
                    "kv_update_tokens": 20,
                    "kv_update_valid_tokens": 16,
                    "kv_update_skipped_tokens": 4,
                    "kv_update_duplicate_slots": 2,
                    "kv_update_valid_token_pct": 80.0,
                    "kv_update_skipped_token_pct": 20.0,
                    "kv_update_duplicate_slot_pct": 12.5,
                },
                "timings": {
                    "decode_step_count": 5,
                    "top_level_total_s": 0.50,
                    "model_execute_total_s": 0.32,
                    "top_level": {
                        "prepare_decode_s": {
                            "records_with_value": 5,
                            "total_s": 0.05,
                            "mean_s": 0.01,
                            "max_s": 0.012,
                        },
                    },
                    "model_execute_subcomponents": {
                        "kv_update_s": {
                            "records_with_value": 5,
                            "total_s": 0.08,
                            "mean_s": 0.016,
                            "max_s": 0.02,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    aggregate = summarize_decode_profile_runs([run_a, run_b])

    assert aggregate["run_count"] == 2
    assert aggregate["timings"]["decode_step_count"] == 8
    assert aggregate["timings"]["top_level_total_s"] == pytest.approx(0.8)
    assert aggregate["timings"]["model_execute_total_s"] == pytest.approx(0.5)
    assert aggregate["counters"]["kv_update_calls"] == 8
    assert aggregate["counters"]["kv_update_tokens"] == 32
    assert aggregate["counters"]["kv_update_valid_tokens"] == 25
    assert aggregate["counters"]["kv_update_skipped_tokens"] == 7
    assert aggregate["counters"]["kv_update_duplicate_slots"] == 3
    assert aggregate["counters"]["kv_update_valid_token_pct"] == pytest.approx(78.125)
    assert aggregate["counters"]["kv_update_duplicate_slot_pct"] == pytest.approx(12.0)
    assert aggregate["timings"]["top_level"]["prepare_decode_s"]["total_s"] == pytest.approx(0.11)
    assert aggregate["timings"]["top_level"]["prepare_decode_s"]["records_with_value"] == 8
    assert aggregate["timings"]["model_execute_subcomponents"]["kv_update_s"]["total_s"] == pytest.approx(0.11)
    assert aggregate["runs"][1]["kv_update_backend"] == "sorted_compact_scatter"
