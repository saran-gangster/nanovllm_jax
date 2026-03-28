# Architecture Brief

## Overview

`nano-vllm-jax` is a local-first JAX inference runtime built around paged KV-cache execution. The codebase is intentionally narrow:

- `src/nanovllm_jax/engine/` owns request scheduling, batching, and model execution flow.
- `src/nanovllm_jax/models/` contains model definitions and weight loading.
- `src/nanovllm_jax/layers/` contains the reusable network blocks plus the paged-attention runtime.
- `src/nanovllm_jax/utils/` contains context, parallel, and loading helpers.

## Runtime Flow

1. `LLM` creates `LLMEngine`.
2. `LLMEngine` batches requests and drives prefill/decode steps.
3. `ModelRunner` prepares padded inputs, owns KV-cache buffers, and executes the model.
4. Attention layers consume the shared `AttentionContext` to choose prefill or decode behavior.

## Attention Stack

The attention implementation is split by responsibility instead of by experiment history:

- [attention.py](src/nanovllm_jax/layers/attention.py): high-level attention layer and KV-cache interaction.
- [paged_attention.py](src/nanovllm_jax/layers/paged_attention.py): runtime dispatch, vectorized/blockwise fallbacks, and Mosaic backend selection.
- [mosaic_gpu_attention.py](src/nanovllm_jax/layers/mosaic_gpu_attention.py): Hopper-oriented Pallas Mosaic kernels.

## Pallas Mosaic GPU Kernels

The Mosaic path is organized around three internal kernel families:

- `baseline`: the most stable batched decode core.
- `latency`: tuned for short-context decode shapes where launch and merge overhead matter more.
- `throughput`: tuned for long-context decode shapes where split-k and scheduling pay off.

These names describe the observed shape regimes from prior H100 work. They are internal dispatch labels, not public API.

## Status

The local runtime and blockwise decode path are the primary supported surfaces today.

The Mosaic GPU kernels remain under active design but are not in an active performance-tuning cycle right now. Prior H100 experiments were promising, but further kernel bring-up, retuning, and benchmark validation are temporarily paused pending renewed access to dedicated Hopper-class GPU hardware.
