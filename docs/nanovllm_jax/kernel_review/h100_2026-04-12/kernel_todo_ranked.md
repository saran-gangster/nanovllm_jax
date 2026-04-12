# Ranked Kernel Work

This is the shortest ranked list of kernel work that still matches the current H100 data.

## 1. Throughput-v2 boundary

Expected payoff: highest

Why:

- current throughput is still a split/broadcast/merge bridge
- current split-k tuning changes behavior but does not close the structural gap
- the H100 data says bridge removal matters more than more geometry retuning

Primary files:

- [src/nanovllm_jax/layers/mosaic_gpu_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/mosaic_gpu_attention.py)
- [src/nanovllm_jax/layers/paged_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/paged_attention.py)

## 2. Dedicated partition reduction boundary

Expected payoff: high

Why:

- current latency and throughput both merge partials in JAX
- that keeps a large structural cost outside the kernel
- even a dedicated reduction kernel would be cleaner than the current wrapper merge

Primary files:

- [src/nanovllm_jax/layers/mosaic_gpu_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/mosaic_gpu_attention.py)
- [tests/test_mosaic_throughput_utils.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/tests/test_mosaic_throughput_utils.py)

## 3. Decode-schedule-owned family metadata

Expected payoff: medium-high

Why:

- schedule ownership is already better than before, but throughput-family metadata is still built around family-specific wrapper structure
- throughput-v2 should consume schedule-owned metadata directly
- better metadata ownership reduces host/JAX prep churn before kernel work even starts

Primary files:

- [src/nanovllm_jax/engine/decode_schedule.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/engine/decode_schedule.py)
- [src/nanovllm_jax/engine/model_runner.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/engine/model_runner.py)
- [src/nanovllm_jax/layers/paged_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/paged_attention.py)

## 4. Strict kernel A/B harness

Expected payoff: medium

Why:

- current one-off benchmarking is enough to orient, not enough to promote
- fresh-process A/B is required for stable kernel claims
- this is cheap to build and removes a lot of argument about noisy results

Primary files:

- [bench_decode_families.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/bench_decode_families.py)
- [bench_decode_kernel_matrix.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/bench_decode_kernel_matrix.py)
- [src/nanovllm_jax/utils/decode_kernel_bench.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/utils/decode_kernel_bench.py)

## 5. KV store kernel v1

Expected payoff: medium

Why:

- the eager H100 micro-attribution still favors `scatter`
- a replacement should not land until it clearly beats the measured baseline
- the instrumentation is now good enough to evaluate a dedicated store kernel honestly

Primary files:

- [src/nanovllm_jax/layers/attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/attention.py)
- [tests/test_kv_update_ops.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/tests/test_kv_update_ops.py)

## What Not To Do First

- broad random tile sweeps
- more tuning of the current throughput bridge without structural changes
- promotion of compact KV scatter variants from non-eager wall clock alone
- family promotion based on one synthetic win without live decode validation
