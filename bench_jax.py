#!/usr/bin/env python3
"""Benchmark: Naive HuggingFace decoding vs Nano-vLLM JAX inference."""

import sys
import os

# Add parent directory to path so 'import nanovllm_jax' works
# When running from /workspace/nanovllm_jax/, we need /workspace/ in path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "/workspace/models/Qwen3-0.6B"
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

print(f"JAX devices: {jax.devices()}")
print(f"Model: {MODEL_PATH}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Number of prompts: {len(PROMPTS)}")
print("=" * 60)

# ============================================================================
# Benchmark 1: Naive HuggingFace PyTorch Decoding
# ============================================================================
print("\n[1] Loading HuggingFace model (PyTorch)...")
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
hf_model.eval()

def naive_hf_generate(prompts, max_new_tokens, temperature):
    """Naive HuggingFace generation (one prompt at a time)."""
    outputs = []
    for prompt in prompts:
        inputs = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        with torch.no_grad():
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=hf_tokenizer.eos_token_id,
            )
        text = hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs

def batched_hf_generate(prompts, max_new_tokens, temperature):
    """Batched HuggingFace generation."""
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    inputs = hf_tokenizer(prompts, return_tensors="pt", padding=True).to(hf_model.device)
    with torch.no_grad():
        output_ids = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=hf_tokenizer.eos_token_id,
        )
    outputs = hf_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs

# Warmup HF
print("Warming up HuggingFace...")
for _ in range(NUM_WARMUP):
    _ = naive_hf_generate(PROMPTS[:1], 10, TEMPERATURE)
    _ = batched_hf_generate(PROMPTS[:1], 10, TEMPERATURE)

# Benchmark naive HF
print("Benchmarking naive HuggingFace (sequential)...")
hf_naive_times = []
for run in range(NUM_RUNS):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    hf_naive_outputs = naive_hf_generate(PROMPTS, MAX_NEW_TOKENS, TEMPERATURE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    hf_naive_times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.3f}s")

# Benchmark batched HF
print("Benchmarking batched HuggingFace...")
hf_batched_times = []
for run in range(NUM_RUNS):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    hf_batched_outputs = batched_hf_generate(PROMPTS, MAX_NEW_TOKENS, TEMPERATURE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    hf_batched_times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.3f}s")

# Calculate HF stats
hf_naive_avg = sum(hf_naive_times) / len(hf_naive_times)
hf_batched_avg = sum(hf_batched_times) / len(hf_batched_times)
hf_naive_tokens = len(PROMPTS) * MAX_NEW_TOKENS
hf_naive_tps = hf_naive_tokens / hf_naive_avg
hf_batched_tps = hf_naive_tokens / hf_batched_avg

print(f"\nHuggingFace Naive: {hf_naive_avg:.3f}s avg, {hf_naive_tps:.1f} tokens/s")
print(f"HuggingFace Batched: {hf_batched_avg:.3f}s avg, {hf_batched_tps:.1f} tokens/s")

# Free HF model memory
del hf_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# Benchmark 2: Nano-vLLM JAX
# ============================================================================
print("\n" + "=" * 60)
print("[2] Loading Nano-vLLM JAX model...")

from nanovllm_jax import LLM, SamplingParams

# Initialize with JIT compilation
nanovllm = LLM(model=MODEL_PATH)
sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)

# Warmup nano-vllm
print("Warming up Nano-vLLM JAX...")
for _ in range(NUM_WARMUP):
    _ = nanovllm.generate(PROMPTS[:1], SamplingParams(temperature=TEMPERATURE, max_tokens=10))

# Benchmark nano-vllm
print("Benchmarking Nano-vLLM JAX...")
nanovllm_times = []
for run in range(NUM_RUNS):
    jax.block_until_ready(jnp.zeros(1))  # Sync JAX
    start = time.perf_counter()
    nanovllm_outputs = nanovllm.generate(PROMPTS, sampling_params)
    jax.block_until_ready(jnp.zeros(1))  # Sync JAX
    elapsed = time.perf_counter() - start
    nanovllm_times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.3f}s")

# Calculate nano-vllm stats
nanovllm_avg = sum(nanovllm_times) / len(nanovllm_times)
nanovllm_tokens = sum(len(out['token_ids']) for out in nanovllm_outputs)
nanovllm_tps = nanovllm_tokens / nanovllm_avg

print(f"\nNano-vLLM JAX: {nanovllm_avg:.3f}s avg, {nanovllm_tps:.1f} tokens/s")

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Method':<25} {'Time (s)':<12} {'Tokens/s':<12} {'Speedup':<10}")
print("-" * 60)
print(f"{'HF Naive (sequential)':<25} {hf_naive_avg:<12.3f} {hf_naive_tps:<12.1f} {'1.00x (baseline)':<10}")
print(f"{'HF Batched':<25} {hf_batched_avg:<12.3f} {hf_batched_tps:<12.1f} {hf_naive_avg/hf_batched_avg:.2f}x")
print(f"{'Nano-vLLM JAX':<25} {nanovllm_avg:<12.3f} {nanovllm_tps:<12.1f} {hf_naive_avg/nanovllm_avg:.2f}x")
print("-" * 60)

# Speedup over batched HF
speedup_vs_batched = hf_batched_avg / nanovllm_avg
print(f"\nNano-vLLM JAX speedup vs HF Batched: {speedup_vs_batched:.2f}x")
print(f"Nano-vLLM JAX speedup vs HF Naive: {hf_naive_avg/nanovllm_avg:.2f}x")

# Show sample outputs
print("\n" + "=" * 60)
print("SAMPLE OUTPUTS (Nano-vLLM JAX)")
print("=" * 60)
for i, (prompt, output) in enumerate(zip(PROMPTS, nanovllm_outputs)):
    print(f"\nPrompt {i+1}: {prompt}")
    print(f"Output: {output['text'][:200]}...")
