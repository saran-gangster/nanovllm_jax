# Comparative Report: PyTorch nano-vllm vs JAX nano-vllm

## Executive Summary

The PyTorch `nanovllm/` implementation achieves higher performance through **CUDA Graphs**, **Triton kernels**, and the **Flash Attention library** with native paged attention support. Our JAX implementation uses XLA compilation and `jax.nn.dot_product_attention`, which is less optimized for paged KV-cache operations.

---

## Benchmark Results

**Test Configuration:**
- Model: Qwen3-0.6B
- 4 prompts × 100 max_new_tokens = 400 total tokens
- Temperature: 0.7
- GPU: NVIDIA RTX A6000 (RunPod)

| Method | Time (s) | Tokens/s | vs HF Naive |
|--------|----------|----------|-------------|
| HuggingFace Naive (sequential) | 9.4 | 42.5 | 1.00x |
| HuggingFace Batched | 3.06 | 130.8 | 3.07x |
| **Nano-vLLM JAX** | **4.09** | **97.9** | **2.30x** |

---

## Key Optimizations Comparison

| Component | PyTorch (nanovllm) | JAX (nanovllm_jax) | Impact |
|-----------|-------------------|-------------------|--------|
| **Graph Capture** | CUDA Graphs with pre-allocated batch sizes [1,2,4,8,16,...,512] | `nnx.jit` with XLA compilation | ⚠️ **High** - CUDA graphs avoid kernel launch overhead |
| **Decode Attention** | `flash_attn_with_kvcache()` with native block table support | Gather K/V → `jax.nn.dot_product_attention` | ⚠️ **High** - Flash Attention is ~2-3x faster for paged decode |
| **Prefill Attention** | `flash_attn_varlen_func()` for variable-length | Padding + masking + `dot_product_attention` | ⚠️ **Medium** - Memory overhead from padding |
| **KV-Cache Store** | Triton kernel (`store_kvcache_kernel`) | `jnp.at[].set()` XLA scatter | ⚡ **Low** - XLA scatter is competitive |
| **RoPE** | `@torch.compile` with Triton | `@jax.jit` | ✅ **Equivalent** |
| **Tensor Parallelism** | NCCL `all_reduce` | `lax.psum` via `shard_map` | ✅ **Equivalent** |
| **Multi-Process** | `SharedMemory` + `multiprocessing` | Single-controller SPMD | ✅ JAX is simpler |

---

## Critical Missing Optimizations in JAX

### 1. CUDA Graphs (High Impact: ~10-30% decode speed)

**PyTorch captures GPU operations into a graph:**
```python
# PyTorch - captures once, replays without CPU overhead
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, self.graph_pool):
    outputs = self.model(input_ids, positions)
# At runtime - just replay
graph.replay()
```

**JAX relies on XLA JIT:**
```python
# JAX - XLA compiles but still has dispatch overhead
logits = self._run_model_jit(self.model, input_ids, positions, context)
```

JAX has no direct equivalent. XLA compilation caches the trace but doesn't eliminate kernel launch overhead like CUDA graphs.

---

### 2. Paged Attention Kernel (High Impact: ~2-3x decode speed)

**PyTorch uses Flash Attention's native paged support:**
```python
# Single optimized kernel that reads directly from block tables
output = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=context.context_lens,
    block_table=context.block_tables,  # Native support!
    softmax_scale=self.scale,
)
```

**JAX gathers K/V into a dense tensor first:**
```python
# Gathers entire K/V (wasteful) then computes attention
k_gathered, v_gathered, mask = gather_kv_from_cache(
    k_cache, v_cache, block_tables, context_lens, block_size
)
output = jax.nn.dot_product_attention(q, k_gathered, v_gathered, mask=mask)
```

This materialization of `[batch, max_blocks * block_size, heads, dim]` is the **biggest performance gap**.

---

### 3. Pre-allocated Graph Variables (Medium Impact: ~5-10%)

**PyTorch pre-allocates tensors for CUDA graphs:**
```python
# Allocated once at startup
self.graph_vars = {
    "input_ids": torch.zeros(max_bs, dtype=torch.int32, device="cuda"),
    "positions": torch.zeros(max_bs, dtype=torch.int32, device="cuda"),
    "slot_mapping": torch.full((max_bs,), -1, dtype=torch.int32, device="cuda"),
}
# At runtime - slice into pre-allocated, no allocation
self.graph_vars["input_ids"][:bs] = input_ids
```

**JAX creates new arrays each step:**
```python
# New array allocation each decode step
input_ids = jnp.asarray(np.array([seq.last_token for seq in seqs]))
```

---

### 4. Triton Kernel for KV-Cache Store (Low Impact)

**PyTorch uses a custom Triton kernel:**
```python
@triton.jit
def store_kvcache_kernel(key_ptr, k_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return  # Skip invalid slots efficiently
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
```

**JAX uses XLA scatter:**
```python
k_cache = k_cache.at[safe_slots].set(key, mode='drop')
```

XLA's scatter is reasonably efficient, so this is a **minor gap**.

---

## Detailed Component Analysis

### Model Runner Comparison

#### PyTorch (`nanovllm/engine/model_runner.py`)
- **CUDA Graph Pool Sharing**: Reuses memory across different batch sizes
- **Pre-allocated Graph Variables**: Fixed tensors that are reused (no allocation at runtime)
- **Batch Size Bucketing**: Compile for fixed sizes [1,2,4,8,16,...,512], use next-largest graph

#### JAX (`nanovllm_jax/engine/model_runner.py`)
- Uses `nnx.jit` for XLA compilation
- No memory pool sharing across batch sizes
- JIT recompiles for new shapes (though XLA caches traces)
- No pre-allocated variable reuse pattern

---

### Attention Layer Comparison

#### PyTorch (`nanovllm/layers/attention.py`)
- **Triton kernel**: Custom fused scatter for KV-cache (handles -1 slots efficiently)
- **Flash Attention with Block Tables**: `flash_attn_with_kvcache` natively supports paged attention
- **Zero-copy prefill with prefix cache**: Direct use of cached K/V

#### JAX (`nanovllm_jax/layers/attention.py`)
- XLA scatter for KV-cache store
- `jax.nn.dot_product_attention` for attention computation
- Gather-based decode attention (less efficient than native paged kernel)
- Variable-length attention uses padding + masking (less memory efficient)

---

### Context Passing

#### PyTorch (Global State)
```python
_CONTEXT = Context()

def set_context(is_prefill, ...):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, ...)

def get_context():
    return _CONTEXT
```

#### JAX (Explicit PyTree)
```python
@dataclass
class AttentionContext:
    is_prefill: bool = False
    cu_seqlens_q: jnp.ndarray | None = None
    # ... registered as PyTree

# Passed explicitly through function calls
logits = model(input_ids, positions, context)
```

JAX approach is more functional/pure but adds PyTree overhead.

---

## Recommendations for JAX Implementation

| Priority | Optimization | Expected Speedup | Effort |
|----------|--------------|------------------|--------|
| **High** | Integrate Pallas/JAX-Triton for paged attention | 30-50% | High |
| **High** | Use `jax-flash-attn` library (if available) | 20-40% | Medium |
| **Medium** | Pre-compile for expected batch sizes | 5-10% | Low |
| **Medium** | Use buffer donation more aggressively | 5-10% | Low |
| **Low** | Explore XLA's "compile once" patterns | TBD | Medium |

### Immediate Improvements

1. **Integrate Pallas for Paged Attention**
   ```python
   # Write a Pallas kernel for paged decode attention
   @pl.kernel
   def paged_attention_kernel(q, k_cache, v_cache, block_tables, ...):
       # Direct paged attention without gathering
   ```

2. **Add Batch Size Bucketing with Pre-compilation**
   ```python
   # Pre-compile for expected batch sizes
   for bs in [1, 2, 4, 8, 16, 32]:
       dummy_input = jnp.zeros((bs, hidden_dim))
       _ = jit_fn(dummy_input)  # Trigger compilation
   ```

3. **Use Buffer Donation More Aggressively**
   ```python
   @partial(jax.jit, donate_argnums=(0, 1))
   def update_cache(k_cache, v_cache, k, v, slots):
       # Donated buffers are reused
       return k_cache.at[slots].set(k), v_cache.at[slots].set(v)
   ```

---

## Summary

| Aspect | PyTorch Advantage | JAX Advantage |
|--------|-------------------|---------------|
| **Decode Speed** | ✅ CUDA graphs, Flash Attention paged kernel | |
| **Prefill Speed** | ✅ Flash Attention varlen | ✅ XLA fusion is competitive |
| **Memory Efficiency** | ✅ Triton scatter, pinned memory | |
| **Multi-GPU** | ✅ Explicit NCCL control | ✅ Simpler SPMD model |
| **Code Simplicity** | | ✅ Functional, no global state |
| **TPU Support** | | ✅ Native TPU backend |

---

## Conclusion

The **~25% performance gap** vs HuggingFace Batched is primarily due to:

1. **No paged attention kernel** - We gather K/V into dense tensors (biggest issue)
2. **No CUDA graphs** - XLA dispatch overhead per step
3. **Python loop overhead** - Scheduling happens in Python each step

To close the gap, the JAX implementation would need a **Pallas-based paged attention kernel** similar to `flash_attn_with_kvcache`. This is a significant implementation effort but would bring performance close to PyTorch.

---

## Optimizations Applied to JAX Version

The following optimizations were implemented during this benchmark session:

1. ✅ **bfloat16 KV Cache** - Changed from float32 to bfloat16 (2x memory bandwidth)
2. ✅ **bfloat16 Model Weights** - Converted all model weights to bfloat16 during loading
3. ✅ **Increased KV Cache Allocation** - From 256MB to 2GB × gpu_memory_utilization
4. ✅ **Fixed Duplicate Return Bug** - Removed duplicate return statements in attention.py
5. ✅ **Optimized KV Cache Gather** - Improved `gather_kv_from_cache` with better indexing
6. ✅ **Optimized KV Cache Store** - Added dtype casting for efficient memory bandwidth

**Result**: Improved from 38.8 tok/s (0.89x HF Naive) to 97.9 tok/s (2.30x HF Naive).
