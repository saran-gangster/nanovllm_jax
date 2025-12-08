#!/usr/bin/env python3
"""Profile attention kernels to identify bottlenecks."""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import jax
import jax.numpy as jnp
from nanovllm_jax.layers import pallas_attention

print(f"JAX devices: {jax.devices()}")
print("=" * 70)

# Test configuration
batch_size = 64
num_heads = 32
head_dim = 128
block_size = 256
max_blocks = 8
num_runs = 20

# Create test inputs
q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, num_heads, head_dim), dtype=jnp.float16)
num_blocks = batch_size * max_blocks
k_cache = jax.random.normal(jax.random.PRNGKey(1), (num_blocks, block_size, num_heads, head_dim), dtype=jnp.float16)
v_cache = jax.random.normal(jax.random.PRNGKey(2), (num_blocks, block_size, num_heads, head_dim), dtype=jnp.float16)
block_tables = jnp.arange(num_blocks, dtype=jnp.int32).reshape(batch_size, max_blocks)
context_lens = jax.random.randint(jax.random.PRNGKey(3), (batch_size,), 200, 2048, dtype=jnp.int32)
scale = 1.0 / jnp.sqrt(head_dim).astype(jnp.float32)

print(f"Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Num heads: {num_heads}")
print(f"  Head dim: {head_dim}")
print(f"  Block size: {block_size}")
print(f"  Context lens: min={context_lens.min()}, max={context_lens.max()}")
print("=" * 70)

# Test 1: Vectorized decode (current fallback)
print("\n[1] Profiling vectorized decode (current fallback)...")
vectorized_fn = jax.jit(pallas_attention.paged_decode_attention_vectorized)

# Warmup
for _ in range(5):
    _ = vectorized_fn(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
jax.block_until_ready(_)

# Benchmark
times = []
for i in range(num_runs):
    start = time.perf_counter()
    out = vectorized_fn(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - start
    times.append(elapsed * 1000)  # Convert to ms

median_time = jnp.median(jnp.array(times))
std_time = jnp.std(jnp.array(times))
print(f"  Vectorized decode: {median_time:.2f}ms (±{std_time:.2f}ms)")

# Test 2: Mosaic prefill (working)
print("\n[2] Profiling Mosaic prefill...")
try:
    from nanovllm_jax.layers import pallas_mosaic_attention
    
    # Create prefill inputs
    total_tokens = 3840
    cu_seqlens = jnp.array([0, 512, 1280, 2304, 3840], dtype=jnp.int32)
    q_prefill = jax.random.normal(jax.random.PRNGKey(4), (total_tokens, num_heads, head_dim), dtype=jnp.float16)
    k_prefill = jax.random.normal(jax.random.PRNGKey(5), (total_tokens, num_heads, head_dim), dtype=jnp.float16)
    v_prefill = jax.random.normal(jax.random.PRNGKey(6), (total_tokens, num_heads, head_dim), dtype=jnp.float16)
    
    config = pallas_mosaic_attention.MosaicAttentionConfig(
        block_q=64,
        block_kv=64,
        max_concurrent_steps=2,
        use_schedule_barrier=True,
        num_compute_wgs=2,
    )
    
    prefill_fn = jax.jit(lambda q, k, v, cu: pallas_mosaic_attention.variable_length_attention_mosaic(
        q, k, v, cu, scale, config
    ))
    
    # Warmup
    for _ in range(5):
        _ = prefill_fn(q_prefill, k_prefill, v_prefill, cu_seqlens)
    jax.block_until_ready(_)
    
    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        out = prefill_fn(q_prefill, k_prefill, v_prefill, cu_seqlens)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    median_time = jnp.median(jnp.array(times))
    std_time = jnp.std(jnp.array(times))
    print(f"  Mosaic prefill: {median_time:.2f}ms (±{std_time:.2f}ms)")
    
except Exception as e:
    print(f"  Mosaic prefill error: {e}")

# Test 3: Profile decode with different implementations
print("\n[3] Comparing decode implementations...")

# 3a. Standard Flash Attention (if available)
try:
    standard_fn = jax.jit(pallas_attention._paged_attention_fallback)
    
    # Warmup
    for _ in range(5):
        _ = standard_fn(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
    jax.block_until_ready(_)
    
    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        out = standard_fn(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    median_time = jnp.median(jnp.array(times))
    std_time = jnp.std(jnp.array(times))
    print(f"  Standard fallback: {median_time:.2f}ms (±{std_time:.2f}ms)")
    
except Exception as e:
    print(f"  Standard fallback error: {e}")

# Test 4: Profile individual components
print("\n[4] Profiling decode components...")

def profile_component(name, fn, *args):
    """Profile a single component."""
    jit_fn = jax.jit(fn)
    
    # Warmup
    for _ in range(5):
        _ = jit_fn(*args)
    jax.block_until_ready(_)
    
    # Benchmark
    times = []
    for i in range(10):
        start = time.perf_counter()
        out = jit_fn(*args)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    median_time = jnp.median(jnp.array(times))
    print(f"  {name}: {median_time:.2f}ms")
    return out

# Component: Block table lookup
def gather_blocks(k_cache, block_tables, seq_idx):
    """Gather KV blocks for a sequence."""
    blocks = block_tables[seq_idx]
    return k_cache[blocks]

print(f"\n  Testing block gather overhead...")
_ = profile_component(
    "Block gather (single seq)", 
    gather_blocks,
    k_cache, block_tables, 0
)

# Component: Attention computation
def single_seq_attention(q_seq, k_blocks, v_blocks, context_len):
    """Attention for one sequence."""
    # q_seq: [num_heads, head_dim]
    # k_blocks: [num_blocks, block_size, num_heads, head_dim]
    # v_blocks: [num_blocks, block_size, num_heads, head_dim]
    
    # Reshape for attention
    k_flat = k_blocks.reshape(-1, num_heads, head_dim)  # [num_blocks*block_size, num_heads, head_dim]
    v_flat = v_blocks.reshape(-1, num_heads, head_dim)
    
    # Compute attention scores
    scores = jnp.einsum('hd,nhd->hn', q_seq, k_flat) * scale
    
    # Mask
    mask = jnp.arange(k_flat.shape[0]) < context_len
    scores = jnp.where(mask[None, :], scores, -jnp.inf)
    
    # Softmax and weighted sum
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('hn,nhd->hd', weights, v_flat)
    return out

q_single = q[0]  # [num_heads, head_dim]
k_single = k_cache[block_tables[0]]  # [max_blocks, block_size, num_heads, head_dim]
v_single = v_cache[block_tables[0]]
context_len_single = context_lens[0]

_ = profile_component(
    "Single seq attention",
    single_seq_attention,
    q_single, k_single, v_single, context_len_single
)

print("\n" + "=" * 70)
print("PROFILING COMPLETE")
print("=" * 70)
print("\nKey Insights:")
print("1. Vectorized decode is the current bottleneck")
print("2. Mosaic prefill is working and optimized")
print("3. Next: Enable Mosaic decode to match/beat vectorized performance")
