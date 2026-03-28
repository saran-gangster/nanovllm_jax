"""Tests for decode backend configuration validation."""

from __future__ import annotations

import pytest

from nanovllm_jax.config import Config
from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.sampling_params import SamplingParams


def test_config_rejects_unknown_decode_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="auto\\|mosaic\\|blockwise"):
        Config(model=str(tmp_path), decode_attention_backend="unknown")


def test_config_rejects_vectorized_decode_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="auto\\|mosaic\\|blockwise"):
        Config(model=str(tmp_path), decode_attention_backend="vectorized")


def test_config_rejects_tensor_parallel_sizes_above_one(tmp_path) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        Config(model=str(tmp_path), tensor_parallel_size=2)


def test_llm_engine_rejects_unknown_config_kwargs(tmp_path) -> None:
    with pytest.raises(TypeError, match="Unexpected LLM config arguments: bogus_flag"):
        LLMEngine(model=str(tmp_path), bogus_flag=True)


def test_sampling_params_reject_near_greedy_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=0.0)
