"""Test script for Mosaic GPU kernel improvements.

Tests:
1. Decode kernel with batching (WGMMA M>=64)
2. Prefill kernel with variable lengths
3. Per-sequence block table handling
4. SMEM budget compliance
"""

import jax
import jax.numpy as jnp
import math
import time

def test_mosaic_decode():
    """Test decode kernel with proper batching."""
    print("\n" + "="*60)
    print("TEST 1: Mosaic Decode Kernel")
    print("="*60)
    
    try:
        from layers.pallas_attention import (
            paged_attention,
            MOSAIC_AVAILABLE,
        )
        
        print(f"Mosaic available: {MOSAIC_AVAILABLE}")
        
        # Test parameters
        batch_size = 64  # WGMMA requires M>=64
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        num_blocks = 128
        block_size = 256
        max_blocks_per_seq = 8
        
        print(f"\nConfiguration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {head_dim}")
        print(f"  Block size: {block_size}")
        print(f"  Max blocks per seq: {max_blocks_per_seq}")
        
        # Create inputs
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, head_dim), dtype=jnp.float16)
        k_cache = jax.random.normal(key, (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.float16)
        v_cache = jax.random.normal(key, (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.float16)
        
        # Create diverse block tables (different sequences use different blocks)
        block_tables = jnp.arange(batch_size * max_blocks_per_seq).reshape(batch_size, max_blocks_per_seq) % num_blocks
        context_lens = jnp.array([200 + i * 50 for i in range(batch_size)], dtype=jnp.int32)
        context_lens = jnp.minimum(context_lens, max_blocks_per_seq * block_size)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        print(f"\nContext lengths: min={context_lens.min()}, max={context_lens.max()}")
        print(f"Block table shape: {block_tables.shape}")
        
        # Warmup
        print("\nWarming up...")
        for _ in range(3):
            _ = paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        for _ in range(10):
            start = time.time()
            out = paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
            out.block_until_ready()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"\n✓ Decode kernel completed successfully")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Output shape: {out.shape}")
        print(f"  Output dtype: {out.dtype}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Decode kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mosaic_prefill():
    """Test prefill kernel with variable sequence lengths."""
    print("\n" + "="*60)
    print("TEST 2: Mosaic Prefill Kernel")
    print("="*60)
    
    try:
        from layers.pallas_attention import (
            paged_prefill_attention,
            MOSAIC_AVAILABLE,
        )
        
        print(f"Mosaic available: {MOSAIC_AVAILABLE}")
        
        # Test parameters
        batch_size = 4
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        seq_lens = [512, 768, 1024, 1536]  # Variable lengths
        
        print(f"\nConfiguration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {head_dim}")
        print(f"  Sequence lengths: {seq_lens}")
        
        # Create inputs
        total_tokens = sum(seq_lens)
        cu_seqlens = jnp.array([0] + [sum(seq_lens[:i+1]) for i in range(batch_size)], dtype=jnp.int32)
        max_seqlen = max(seq_lens)
        
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (total_tokens, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        print(f"\nTotal tokens: {total_tokens}")
        print(f"Max sequence length: {max_seqlen}")
        print(f"cu_seqlens: {cu_seqlens}")
        
        # Warmup
        print("\nWarming up...")
        for _ in range(3):
            _ = paged_prefill_attention(q, k, v, cu_seqlens, max_seqlen, scale)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        for _ in range(10):
            start = time.time()
            out = paged_prefill_attention(q, k, v, cu_seqlens, max_seqlen, scale)
            out.block_until_ready()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"\n✓ Prefill kernel completed successfully")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Output shape: {out.shape}")
        print(f"  Output dtype: {out.dtype}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Prefill kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smem_budget():
    """Test SMEM budget compliance."""
    print("\n" + "="*60)
    print("TEST 3: SMEM Budget Compliance")
    print("="*60)
    
    try:
        from layers.pallas_mosaic_attention import (
            _cap_pipeline_depth,
            _SMEM_BUDGET_BYTES,
        )
        
        print(f"SMEM budget: {_SMEM_BUDGET_BYTES / 1024:.1f} KB")
        
        # Test various configurations
        configs = [
            {"block_q": 64, "block_kv": 64, "head_dim": 128, "requested": 2},
            {"block_q": 64, "block_kv": 64, "head_dim": 128, "requested": 4},
            {"block_q": 64, "block_kv": 128, "head_dim": 128, "requested": 2},
            {"block_q": 128, "block_kv": 64, "head_dim": 128, "requested": 2},
        ]
        
        print("\nTesting configurations:")
        all_passed = True
        
        for cfg in configs:
            try:
                capped = _cap_pipeline_depth(
                    block_q=cfg["block_q"],
                    block_kv=cfg["block_kv"],
                    head_dim=cfg["head_dim"],
                    dtype=jnp.float16,
                    num_compute_wgs=2,
                    requested_steps=cfg["requested"],
                    metadata_bytes=4096,  # Example metadata size
                )
                
                status = "✓" if capped > 0 else "✗"
                print(f"  {status} block_q={cfg['block_q']}, block_kv={cfg['block_kv']}, "
                      f"requested={cfg['requested']} -> capped={capped}")
                
                if capped == 0:
                    all_passed = False
                    
            except ValueError as e:
                print(f"  ✗ block_q={cfg['block_q']}, block_kv={cfg['block_kv']}: {e}")
                all_passed = False
        
        if all_passed:
            print(f"\n✓ All SMEM budget tests passed")
        else:
            print(f"\n✗ Some SMEM budget tests failed")
            
        return all_passed
        
    except Exception as e:
        print(f"\n✗ SMEM budget test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_per_sequence_blocks():
    """Test per-sequence block table handling."""
    print("\n" + "="*60)
    print("TEST 4: Per-Sequence Block Table Handling")
    print("="*60)
    
    try:
        from layers.pallas_attention import paged_attention
        
        # Small test with different block patterns
        batch_size = 4
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        num_blocks = 16
        block_size = 64
        max_blocks_per_seq = 4
        
        print(f"\nConfiguration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Block size: {block_size}")
        
        # Create inputs with different block patterns per sequence
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, head_dim), dtype=jnp.float16)
        k_cache = jax.random.normal(key, (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.float16)
        v_cache = jax.random.normal(key, (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.float16)
        
        # Different block patterns for each sequence
        block_tables = jnp.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ], dtype=jnp.int32)
        
        context_lens = jnp.array([100, 150, 200, 250], dtype=jnp.int32)
        scale = 1.0 / math.sqrt(head_dim)
        
        print(f"Block tables:\n{block_tables}")
        print(f"Context lengths: {context_lens}")
        
        # Run attention
        print("\nRunning attention...")
        out = paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
        
        print(f"\n✓ Per-sequence block handling completed")
        print(f"  Output shape: {out.shape}")
        print(f"  Output mean: {out.mean():.6f}")
        print(f"  Output std: {out.std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Per-sequence block test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MOSAIC GPU KERNEL TEST SUITE")
    print("="*60)
    
    # Check JAX devices
    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    
    # Run tests
    results = {
        "decode": test_mosaic_decode(),
        "prefill": test_mosaic_prefill(),
        "smem_budget": test_smem_budget(),
        "per_sequence": test_per_sequence_blocks(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
