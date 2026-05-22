#!/usr/bin/env python3
"""Smoke test for the nano-vLLM JAX implementation.

This is intentionally a lightweight "does it run?" script (not a unit test).

Set `NANOVLLM_MODEL_PATH` to a local HuggingFace model directory to override
the default repo-local path.
"""

from __future__ import annotations

import os
from pathlib import Path

from nanovllm_jax import LLM, SamplingParams

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "models/qwen/Qwen-3-0.6B"
    model_path = Path(os.environ.get("NANOVLLM_MODEL_PATH", str(default_model))).expanduser()

    if not model_path.exists():
        raise SystemExit(
            f"Model path does not exist: {model_path}\n"
            "Set NANOVLLM_MODEL_PATH to a local HuggingFace model directory."
        )

    # For single GPU (tensor_parallel_size=1 is default)
    llm = LLM(model=str(model_path))
    prompts = ["Hello, my name is"]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=100))

    print("Generated outputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
