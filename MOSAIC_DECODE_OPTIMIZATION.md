# Mosaic Decode Optimization Plan

## Current Status (From Profiling)

**Benchmark Results:**
- HF Batched: 127.5 tok/s (3.137s median)
- Nano-vLLM JAX: 115.1 tok/s (3.475s median) 
- **Gap: 10% slower** (~12 tok/s difference)

**Profiling Results (H100):**
- Vectorized decode (current): **3.24ms** ±0.05ms
- Block gather overhead: 0.60ms
- Single sequence attention: 0.22ms
- **Total decode dominates end-to-end performance**

## Problem: 1D Iota Layout Inference

Mosaic GPU requires explicit layouts for all operations. For 1D index arrays:

❌ **No layout**: `Failed to infer output layout`
❌ **WGMMA layout**: `Tiling does not apply to shape (64,1)` 
❌ **WG_SPLAT layout**: `NotImplementedError in thread_idxs`

## Solution: Pre-compute Index Arrays on Host

### Key Insight
**Don't compute indices inside kernel** - pass them as inputs!

```python
# HOST: Pre-compute (no layout constraints)
row_ids = jnp.arange(block_q, dtype=jnp.int32)  # [block_q]
col_ids = jnp.arange(block_kv, dtype=jnp.int32)  # [block_kv]

# KERNEL: Pass as inputs, use directly
def kernel(q_ref, k_ref, v_ref, row_ids_ref, col_ids_ref, out_ref):
    row_indices = row_ids_ref[:]  # Simple lookup, no layout issues!
    col_indices = col_ids_ref[:]
    # ... use for masking/indexing
```

### Implementation Plan

1. **Modify `batched_decode_attention_mosaic` signature:**
   ```python
   def batched_decode_attention_mosaic(
       q, k_cache, v_cache, block_tables, context_lens, scale, config,
       # NEW: Pre-computed indices
       row_ids=None,  # [block_q]
       col_ids=None,  # [block_kv]
   ):
       if row_ids is None:
           row_ids = jnp.arange(config.block_q, dtype=jnp.int32)
       if col_ids is None:
           col_ids = jnp.arange(config.block_kv, dtype=jnp.int32)
   ```

2. **Pass to kernel as inputs:**
   - Add `row_ids_ref` and `col_ids_ref` parameters
   - Pass in `plgpu.kernel()` call

3. **Replace all `broadcasted_iota` calls:**
   - Line 594: `row_ids = row_ids_ref[:]`
   - Line 601: `row_indices = row_ids_ref[:]`
   - Line 668: `col_ids = col_ids_ref[:]`
   - Line 1013: `row_ids = row_ids_ref[:]`
   - Line 1389: `seq_indices = row_ids_ref[:] + batch_start`
   - Line 1428: `kv_positions = col_ids_ref[:] + offset`

## Expected Performance Gain

**Current bottleneck:** Vectorized decode = 3.24ms

**Mosaic decode potential:**
- Warp-specialized scheduling: ~15-20% faster
- Better memory coalescing: ~10-15% faster
- WGMMA acceleration: ~20-30% faster
- **Estimated total: 2.2-2.5ms** (25-30% improvement)

**Impact on end-to-end:**
- Current: 3.475s (115.1 tok/s)
- With Mosaic decode: **~3.1-3.2s (~125-130 tok/s)**
- **Target: Match or beat HF Batched (127.5 tok/s)** ✓

## Validation Plan

1. **Test pre-computed indices approach** (mosaic_decode_precomputed_indices.py)
2. **Apply to full decode kernel** (pallas_mosaic_attention.py)
3. **Re-enable dispatch** (pallas_attention.py)
4. **Run tests** (test_mosaic_kernels.py - all 4 should pass)
5. **Benchmark** (bench_jax.py - should close the 10% gap)
6. **Profile again** (profile_attention.py - confirm Mosaic decode is faster)

## Additional Optimizations (After Mosaic Decode)

1. **Prefill scheduling**: Currently uses default config, could be tuned
2. **Mixed batch handling**: Better separation of prefill/decode
3. **Block allocation**: Could pre-allocate more aggressively
4. **Memory layout**: K/V cache layout optimization
5. **JIT compilation**: Reduce variance in first runs

## Success Criteria

✅ All 4 tests passing with Mosaic decode enabled
✅ Mosaic decode < 2.5ms (vs 3.24ms vectorized)
✅ End-to-end ≥ 127 tok/s (match HF Batched)
✅ Stable performance (std < 0.05s)
