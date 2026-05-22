# Architecture

`nano-vLLM JAX` is organized as a small local inference runtime. The code is
split by ownership boundary rather than by experiment history.

## Package Layout

- `nanovllm_jax.llm`: public `LLM` entry point
- `nanovllm_jax.config`: runtime configuration and validation
- `nanovllm_jax.engine`: sequence state, scheduling, block allocation, model runs
- `nanovllm_jax.models`: model definitions and checkpoint loading
- `nanovllm_jax.layers`: reusable model layers and paged attention dispatch
- `nanovllm_jax.utils`: context helpers, runtime diagnostics, profiling helpers

## Runtime Flow

1. `LLM` constructs an `LLMEngine`.
2. `LLMEngine` owns request lifecycle and scheduling.
3. `ModelRunner` prepares padded model inputs and owns KV-cache buffers.
4. Attention layers consume the active `AttentionContext`.
5. The sampler converts logits into next-token choices.

## Attention Boundary

Paged attention is centered in `src/nanovllm_jax/layers/paged_attention.py`.
The portable blockwise path is the stable fallback. Hopper Pallas Mosaic decode
kernels live in `src/nanovllm_jax/layers/mosaic_gpu_attention.py` and are kept
behind runtime gates because they require GPU-specific validation.

The public runtime should stay conservative: CPU behavior and portable decode
must remain correct even when GPU-specific kernels are disabled.

## Development Rule

Keep new features local to the subsystem they affect. For performance work,
preserve a pure-JAX fallback, add CPU-safe tests for dispatch and shape logic,
and validate GPU kernels separately on a matching accelerator.
