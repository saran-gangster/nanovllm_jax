#!/usr/bin/env python3
"""Small end-to-end generation check for the local JAX runtime."""

from __future__ import annotations

import os
from pathlib import Path

from nanovllm_jax import LLM, SamplingParams


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_model = repo_root / "models/qwen/Qwen-3-0.6B"
    model_path = Path(os.environ.get("NANOVLLM_MODEL_PATH", str(default_model))).expanduser()

    if not model_path.exists():
        raise SystemExit(
            f"Model path does not exist: {model_path}\n"
            "Set NANOVLLM_MODEL_PATH to a local HuggingFace model directory."
        )

    llm = LLM(model=str(model_path))
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0.7, max_tokens=100),
    )

    print("Generated outputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
