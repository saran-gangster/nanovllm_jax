#!/usr/bin/env python3
"""Benchmark: Naive HuggingFace decoding vs Nano-vLLM JAX inference.

Handles variance by:
1. More warmup iterations
2. Using median instead of mean
3. Filtering outliers (JIT recompilation spikes)
4. Reporting confidence intervals
"""

import sys
import os

# Add parent directory to path so 'import nanovllm_jax' works
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM
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
NUM_WARMUP = 5  # More warmup for JIT stability
NUM_RUNS = 10   # More runs for better statistics

def compute_stats(times, name=""):
    """Compute statistics with outlier filtering."""
    times = np.array(times)
    
    # Filter outliers using IQR method
    q1, q3 = np.percentile(times, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = times[(times >= lower) & (times <= upper)]
    
    if len(filtered) < len(times) * 0.5:
        # If too many filtered, use all
        filtered = times
    
    median = np.median(filtered)
    mean = np.mean(filtered)
    std = np.std(filtered)
    p10 = np.percentile(filtered, 10)
    p90 = np.percentile(filtered, 90)
    
    return {
        'median': median,
        'mean': mean,
        'std': std,
        'p10': p10,
        'p90': p90,
        'n_filtered': len(times) - len(filtered),
        'n_total': len(times),
    }

print(f"JAX devices: {jax.devices()}")
print(f"Model: {MODEL_PATH}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Number of prompts: {len(PROMPTS)}")
print(f"Warmup runs: {NUM_WARMUP}, Benchmark runs: {NUM_RUNS}")
print("=" * 70)

# ============================================================================
# Benchmark 1: HuggingFace PyTorch
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
hf_naive_stats = compute_stats(hf_naive_times)
hf_batched_stats = compute_stats(hf_batched_times)
total_tokens = len(PROMPTS) * MAX_NEW_TOKENS

hf_naive_tps = total_tokens / hf_naive_stats['median']
hf_batched_tps = total_tokens / hf_batched_stats['median']

print(f"\nHuggingFace Naive:   {hf_naive_stats['median']:.3f}s median (±{hf_naive_stats['std']:.3f}s), {hf_naive_tps:.1f} tok/s")
print(f"HuggingFace Batched: {hf_batched_stats['median']:.3f}s median (±{hf_batched_stats['std']:.3f}s), {hf_batched_tps:.1f} tok/s")

# Free HF model memory
del hf_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# Benchmark 2: Nano-vLLM JAX
# ============================================================================
print("\n" + "=" * 70)
print("[2] Loading Nano-vLLM JAX model...")

from nanovllm_jax import LLM, SamplingParams

# Initialize with JIT compilation
nanovllm = LLM(model=MODEL_PATH)
sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)

# Heavy warmup for JIT stability
print("Warming up Nano-vLLM JAX (heavy warmup for JIT stability)...")
for i in range(NUM_WARMUP * 2):  # Double warmup for JAX
    _ = nanovllm.generate(PROMPTS, SamplingParams(temperature=TEMPERATURE, max_tokens=20), use_tqdm=False)
    if i == 0:
        print(f"  Initial JIT compilation complete")

# Benchmark nano-vllm
print(f"Benchmarking Nano-vLLM JAX ({NUM_RUNS} runs)...")
nanovllm_times = []
for run in range(NUM_RUNS):
    # Ensure JAX is synced before timing
    jax.block_until_ready(jnp.zeros(1))
    start = time.perf_counter()
    nanovllm_outputs = nanovllm.generate(PROMPTS, sampling_params, use_tqdm=False)
    # Block until all computation is done
    jax.block_until_ready(jnp.zeros(1))
    elapsed = time.perf_counter() - start
    nanovllm_times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.3f}s")

# Calculate nano-vllm stats
nanovllm_stats = compute_stats(nanovllm_times)
nanovllm_tokens = sum(len(out['token_ids']) for out in nanovllm_outputs)
nanovllm_tps = nanovllm_tokens / nanovllm_stats['median']

print(f"\nNano-vLLM JAX: {nanovllm_stats['median']:.3f}s median (±{nanovllm_stats['std']:.3f}s), {nanovllm_tps:.1f} tok/s")
if nanovllm_stats['n_filtered'] > 0:
    print(f"  (Filtered {nanovllm_stats['n_filtered']} outliers from {nanovllm_stats['n_total']} runs)")

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY (using median for robustness)")
print("=" * 70)
print(f"{'Method':<25} {'Median (s)':<12} {'Std (s)':<10} {'Tok/s':<12} {'Speedup':<10}")
print("-" * 70)
print(f"{'HF Naive (sequential)':<25} {hf_naive_stats['median']:<12.3f} {hf_naive_stats['std']:<10.3f} {hf_naive_tps:<12.1f} {'1.00x (baseline)':<10}")
print(f"{'HF Batched':<25} {hf_batched_stats['median']:<12.3f} {hf_batched_stats['std']:<10.3f} {hf_batched_tps:<12.1f} {hf_naive_stats['median']/hf_batched_stats['median']:.2f}x")
print(f"{'Nano-vLLM JAX':<25} {nanovllm_stats['median']:<12.3f} {nanovllm_stats['std']:<10.3f} {nanovllm_tps:<12.1f} {hf_naive_stats['median']/nanovllm_stats['median']:.2f}x")
print("-" * 70)

# Speedup comparisons
speedup_vs_batched = hf_batched_stats['median'] / nanovllm_stats['median']
print(f"\nNano-vLLM JAX vs HF Batched: {speedup_vs_batched:.2f}x {'(faster)' if speedup_vs_batched > 1 else '(slower)'}")
print(f"Nano-vLLM JAX vs HF Naive:   {hf_naive_stats['median']/nanovllm_stats['median']:.2f}x (faster)")

# Detailed stats
print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)
print(f"{'Method':<25} {'P10 (s)':<10} {'Median (s)':<12} {'P90 (s)':<10} {'Range':<15}")
print("-" * 70)
print(f"{'HF Naive':<25} {hf_naive_stats['p10']:<10.3f} {hf_naive_stats['median']:<12.3f} {hf_naive_stats['p90']:<10.3f} {hf_naive_stats['p90']-hf_naive_stats['p10']:.3f}s")
print(f"{'HF Batched':<25} {hf_batched_stats['p10']:<10.3f} {hf_batched_stats['median']:<12.3f} {hf_batched_stats['p90']:<10.3f} {hf_batched_stats['p90']-hf_batched_stats['p10']:.3f}s")
print(f"{'Nano-vLLM JAX':<25} {nanovllm_stats['p10']:<10.3f} {nanovllm_stats['median']:<12.3f} {nanovllm_stats['p90']:<10.3f} {nanovllm_stats['p90']-nanovllm_stats['p10']:.3f}s")

# Show sample outputs
print("\n" + "=" * 70)
print("SAMPLE OUTPUTS (Nano-vLLM JAX)")
print("=" * 70)
for i, (prompt, output) in enumerate(zip(PROMPTS, nanovllm_outputs)):
    print(f"\nPrompt {i+1}: {prompt}")
    print(f"Output: {output['text'][:200]}...")
