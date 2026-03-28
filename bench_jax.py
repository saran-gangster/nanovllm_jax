#!/usr/bin/env python3
"""Benchmark Nano-vLLM JAX end-to-end throughput.

This repo is JAX-only: this benchmark intentionally avoids PyTorch baselines
so it can run in environments without `torch` installed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import jax

from nanovllm_jax import LLM, SamplingParams

# =========================================================================
# Configuration
# =========================================================================
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _REPO_ROOT / "models/qwen/Qwen-3-0.6B"
MODEL_PATH = os.environ.get("NANOVLLM_MODEL_PATH", str(_DEFAULT_MODEL_PATH))

PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is",
    "The theory of relativity states that",
    "Python is a programming language that",
]

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
NUM_WARMUP = 2
NUM_RUNS = 5


def main() -> None:
    print(f"JAX devices: {jax.devices()}")
    print(f"Model: {MODEL_PATH}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Number of prompts: {len(PROMPTS)}")
    print("=" * 60)

    llm = LLM(model=MODEL_PATH)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)

    print("Warming up Nano-vLLM JAX...")
    for _ in range(NUM_WARMUP):
        _ = llm.generate(
            PROMPTS[:1],
            SamplingParams(temperature=TEMPERATURE, max_tokens=10),
            use_tqdm=False,
        )

    print("Benchmarking Nano-vLLM JAX...")
    times: list[float] = []
    outputs = None
    for run in range(NUM_RUNS):
        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s")

    assert outputs is not None
    total_tokens = sum(len(out["token_ids"]) for out in outputs)
    avg = sum(times) / len(times)
    tps = total_tokens / avg
    print(f"\nNano-vLLM JAX: {avg:.3f}s avg, {tps:.1f} tokens/s")

    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS (Nano-vLLM JAX)")
    print("=" * 60)
    for i, (prompt, output) in enumerate(zip(PROMPTS, outputs)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Output: {output['text'][:200]}...")


if __name__ == "__main__":
    main()
