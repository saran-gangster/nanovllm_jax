"""Profile decode path to find exact bottlenecks."""

import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm_jax import LLM, SamplingParams
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.utils.context import create_decode_context


def profile_component(name, func, *args, warmup=3, runs=20, **kwargs):
    """Profile a single component."""
    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()
    
    # Timed runs
    times = []
    for _ in range(runs):
        jax.block_until_ready(jax.device_put(0))  # Sync
        start = perf_counter()
        result = func(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()
        times.append(perf_counter() - start)
    
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  {name}: {avg_ms:.3f} ± {std_ms:.3f} ms")
    return avg_ms


def main():
    print("=" * 60)
    print("DECODE PATH PROFILING")
    print("=" * 60)
    
    model_path = "/workspace/models/Qwen3-0.6B"
    
    print("\n[1] Loading model...")
    llm = LLM(model=model_path)
    runner = llm.model_runner
    model = runner.model
    sampler = runner.sampler
    
    # Get model config
    config = runner.config
    hf_config = config.hf_config
    print(f"  Model: {model_path}")
    print(f"  Hidden size: {hf_config.hidden_size}")
    print(f"  Num layers: {hf_config.num_hidden_layers}")
    print(f"  Num heads: {hf_config.num_attention_heads}")
    print(f"  Num KV heads: {hf_config.num_key_value_heads}")
    
    # Create test scenario: 4 sequences, each with ~100 tokens context
    batch_size = 4
    context_len = 100
    block_size = runner.block_size
    
    print(f"\n[2] Setting up test scenario...")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_len}")
    print(f"  Block size: {block_size}")
    
    # Create dummy sequences
    seqs = []
    for i in range(batch_size):
        seq = Sequence(list(range(context_len)))
        seq.temperature = 0.7
        # Assign blocks
        num_blocks = (context_len + block_size - 1) // block_size
        seq.block_table = list(range(i * num_blocks, (i + 1) * num_blocks))
        seqs.append(seq)
    
    # Prepare decode inputs using runner's method
    input_ids, positions, context = runner._prepare_decode(seqs)
    temperatures = runner._prepare_sample(seqs)
    
    print(f"\n[3] Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  positions: {positions.shape}")
    print(f"  block_tables: {context.block_tables.shape}")
    print(f"  context_lens: {context.context_lens}")
    
    # =========================================================================
    # Profile individual components
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPONENT PROFILING (20 runs each)")
    print("=" * 60)
    
    # 1. Full decode step (model + sample)
    print("\n[A] Full decode step:")
    
    def full_decode():
        logits = runner._run_model(input_ids, positions, context, is_prefill=False)
        tokens = sampler(logits, temperatures)
        return tokens
    
    full_time = profile_component("Full decode", full_decode)
    
    # 2. Model forward only
    print("\n[B] Model forward (without sampling):")
    
    def model_forward():
        return runner._run_model(input_ids, positions, context, is_prefill=False)
    
    model_time = profile_component("Model forward", model_forward)
    
    # 3. Sampling only
    print("\n[C] Sampling only:")
    logits = runner._run_model(input_ids, positions, context, is_prefill=False)
    
    def sample_only():
        return sampler(logits, temperatures)
    
    sample_time = profile_component("Sampling", sample_only)
    
    # 4. Embedding lookup
    print("\n[D] Embedding lookup:")
    
    def embed_only():
        return model.model.embed_tokens(input_ids)
    
    embed_time = profile_component("Embedding", embed_only)
    
    # 5. Single decoder layer
    print("\n[E] Single decoder layer:")
    hidden = model.model.embed_tokens(input_ids)
    residual = jnp.zeros_like(hidden)
    layer0 = model.model.layers[0]
    
    def single_layer():
        h, r = layer0(positions, hidden, residual, context)
        return h
    
    layer_time = profile_component("Single layer", single_layer)
    
    # 6. Attention in single layer (need to break down further)
    print("\n[F] Attention breakdown:")
    
    # Get attention module
    attn = layer0.self_attn
    
    # Prepare input for attention
    h_norm, _ = layer0.input_layernorm._add_rms_forward(hidden, residual)
    
    # QKV projection
    def qkv_proj():
        return attn.qkv_proj(h_norm)
    
    qkv_time = profile_component("QKV projection", qkv_proj)
    
    # Full attention forward
    def attn_forward():
        return attn(positions, h_norm, context)
    
    attn_time = profile_component("Full attention", attn_forward)
    
    # 7. MLP
    print("\n[G] MLP:")
    mlp = layer0.mlp
    
    def mlp_forward():
        return mlp(h_norm)
    
    mlp_time = profile_component("MLP forward", mlp_forward)
    
    # 8. LM Head
    print("\n[H] LM Head:")
    # Get last hidden state
    def lm_head():
        return model.lm_head(hidden)
    
    lm_head_time = profile_component("LM Head", lm_head)
    
    # =========================================================================
    # Profile KV cache operations specifically
    # =========================================================================
    print("\n" + "=" * 60)
    print("KV CACHE OPERATIONS")
    print("=" * 60)
    
    from nanovllm_jax.layers.attention import gather_kv_from_cache, store_kv_to_cache
    
    # Get cache from first attention layer
    attn0 = model.model.layers[0].self_attn.attn
    k_cache = attn0.k_cache
    v_cache = attn0.v_cache
    
    print(f"\n  KV cache shape: {k_cache.shape}")
    print(f"  KV cache dtype: {k_cache.dtype}")
    
    # Profile gather
    def gather_kv():
        return gather_kv_from_cache(
            k_cache, v_cache,
            context.block_tables,
            context.context_lens,
            block_size
        )
    
    gather_time = profile_component("KV gather", gather_kv)
    
    # Skip store profiling - causes buffer donation issues
    store_time = 0.0
    print("  KV store: (skipped - buffer donation)")
    
    # Profile dot product attention
    print("\n[I] Dot product attention:")
    k_gathered, v_gathered, mask = gather_kv_from_cache(
        k_cache, v_cache, context.block_tables, context.context_lens, block_size
    )
    
    # Prepare Q
    head_dim_attn = k_cache.shape[3]
    q_dummy = jnp.ones((batch_size, 1, hf_config.num_attention_heads // 1, head_dim_attn), dtype=jnp.bfloat16)
    mask_expanded = mask[:, None, None, :]
    
    def dot_product_attn():
        return jax.nn.dot_product_attention(
            q_dummy, k_gathered, v_gathered,
            mask=mask_expanded,
            scale=head_dim_attn ** -0.5
        )
    
    dpa_time = profile_component("dot_product_attention", dot_product_attn)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nPer-token decode breakdown:")
    print(f"  Full decode:     {full_time:.3f} ms")
    print(f"  ├─ Model:        {model_time:.3f} ms ({100*model_time/full_time:.1f}%)")
    print(f"  │  ├─ Embed:     {embed_time:.3f} ms")
    print(f"  │  ├─ Layers:    {layer_time * hf_config.num_hidden_layers:.3f} ms (est.)")
    print(f"  │  │  ├─ Attn:   {attn_time:.3f} ms/layer")
    print(f"  │  │  │  ├─ QKV: {qkv_time:.3f} ms")
    print(f"  │  │  │  ├─ KV gather: {gather_time:.3f} ms")
    print(f"  │  │  │  └─ DPA: {dpa_time:.3f} ms")
    print(f"  │  │  └─ MLP:    {mlp_time:.3f} ms/layer")
    print(f"  │  └─ LM Head:   {lm_head_time:.3f} ms")
    print(f"  └─ Sampling:     {sample_time:.3f} ms ({100*sample_time/full_time:.1f}%)")
    
    # Calculate theoretical throughput
    tokens_per_sec = batch_size / (full_time / 1000)
    print(f"\nTheoretical decode throughput: {tokens_per_sec:.1f} tok/s")
    print(f"(Batch size {batch_size}, {full_time:.3f} ms per step)")
    
    # Identify bottleneck
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    components = [
        ("Embedding", embed_time),
        ("Attention (per layer)", attn_time),
        ("MLP (per layer)", mlp_time),
        ("LM Head", lm_head_time),
        ("Sampling", sample_time),
        ("KV Gather", gather_time),
        ("Dot Product Attn", dpa_time),
    ]
    
    components.sort(key=lambda x: x[1], reverse=True)
    print("\nComponents by time (descending):")
    for name, time_ms in components:
        print(f"  {name}: {time_ms:.3f} ms")


if __name__ == "__main__":
    main()
