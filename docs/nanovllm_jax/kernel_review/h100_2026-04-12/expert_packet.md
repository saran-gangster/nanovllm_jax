# H100 Mosaic Decode Kernel Expert Packet

This packet is for a field expert review of the current JAX + Pallas Mosaic GPU decode work in this repository.

The specific request is:

- identify the next best technical step
- identify the highest-value kernel boundary to replace next
- describe how the new kernel should be written in Pallas Mosaic GPU
- use the measurements and code references here, not generic FlashAttention advice

## Update After The Initial Packet

This branch moved after the initial packet was prepared.

The most important follow-up result on April 12, 2026 is:

- a new env-gated Pallas Mosaic `throughput_v2` partials kernel now exists behind the same `throughput_v2` boundary
- on a later H100 NVL session, that kernel beat `blockwise` on the full six-shape primary synthetic gate
- the same path stayed within roughly `1e-3` max abs diff against `blockwise` on all six validated rows
- a short controlled real-model runtime probe also moved in the same direction

That means the original expert recommendation to replace the old bridge with a different kernel boundary was correct, and the branch now contains an initial implementation of that recommendation.

The authoritative follow-up result note is now:

- [results_h100_2026-04-12.md](./results_h100_2026-04-12.md)

New tracked artifacts added after the initial packet:

- [artifacts/throughput_v2_mosaic_matrix_summary.json](./artifacts/throughput_v2_mosaic_matrix_summary.json)
- [artifacts/throughput_v2_mosaic_verify_summary.json](./artifacts/throughput_v2_mosaic_verify_summary.json)
- [artifacts/throughput_v2_runtime_blockwise_summary.json](./artifacts/throughput_v2_runtime_blockwise_summary.json)
- [artifacts/throughput_v2_runtime_mosaic_summary.json](./artifacts/throughput_v2_runtime_mosaic_summary.json)

## What We Need From The Expert

Please answer these questions directly:

1. Based on the measured H100 data, what is the next best step?
   - `throughput_v2` bridge removal
   - dedicated partition-reduction kernel
   - schedule-owned metadata refactor
   - a different kernel boundary entirely
2. If you were writing the next decode kernel in Pallas Mosaic GPU, what exact decomposition would you choose?
   - one kernel or two kernels
   - how K/V partitioning should be represented
   - where reduction should happen
   - whether the current batched core should be reused or bypassed
   - whether TMA staging and WGMMA tiling should be reorganized
3. Should we target long-context throughput first, or first replace the current JAX-side partial merge?
4. What is the minimal benchmark matrix that should gate promotion of the next kernel?
5. Which current assumptions in this branch are likely wrong, low-leverage, or misleading?

## Current Branch Contents

The current branch contains four major pieces:

1. Decode runtime diagnostics and schedule ownership work
2. GPU measurement tooling and summary comparison tooling
3. `throughput_v2` scaffolding as a non-default runtime seam
4. Tracked H100 benchmark notes and copied JSON artifacts

Primary code entrypoints to inspect:

- [mosaic_gpu_attention.py](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)
- [paged_attention.py](../../../../src/nanovllm_jax/layers/paged_attention.py)
- [decode_schedule.py](../../../../src/nanovllm_jax/engine/decode_schedule.py)
- [model_runner.py](../../../../src/nanovllm_jax/engine/model_runner.py)
- [attention.py](../../../../src/nanovllm_jax/layers/attention.py)

Primary benchmark/tooling entrypoints:

- [bench_decode_families.py](../../../../bench_decode_families.py)
- [bench_decode_kernel_matrix.py](../../../../bench_decode_kernel_matrix.py)
- [profile_decode_runtime.py](../../../../profile_decode_runtime.py)
- [profile_kv_update_backends.py](../../../../profile_kv_update_backends.py)

Primary review notes:

- [kernel_development_considerations.md](./kernel_development_considerations.md)
- [results_h100_2026-04-12.md](./results_h100_2026-04-12.md)
- [throughput_v2_checklist.md](./throughput_v2_checklist.md)
- [kernel_todo_ranked.md](./kernel_todo_ranked.md)

## Measured Environment

GPU and software:

- GPU: `NVIDIA H100 80GB HBM3`
- Driver: `580.126.09`
- Python: `3.11.10`
- JAX: `0.9.2`
- JAX backend: `gpu`
- JAX device: `CudaDevice(id=0)`

Sample live `nvidia-smi` while tuning:

- P-state: `P0`
- graphics clock: `1980 MHz`
- SM clock: `1980 MHz`
- memory clock: `2619 MHz`
- power draw: `164.90 W / 700.00 W`
- temperature: `38 C`
- memory use: `61373 MiB / 81559 MiB`

## What The Measurements Say

### 1. `blockwise` is still the only competitive decode family

Direct family sweep summaries:

- [family_sweep_summary.json](./artifacts/family_sweep_summary.json)
- [extended_sweeps_summary.json](./artifacts/extended_sweeps_summary.json)

Selected means in milliseconds:

| Shape | Blockwise | Baseline | Latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| `batch=64, head_dim=128, blocks=16` | `1.56` | `135.58` | `147.35` | `151.99` |
| `batch=512, head_dim=128, blocks=16` | `9.45` | `140.94` | `144.41` | `150.94` |
| `batch=4096, head_dim=128, blocks=16` | `72.15` | `172.42` | `181.77` | `185.39` |
| `batch=512, head_dim=128, blocks=64` | `37.10` | `147.41` | `169.88` | `150.32` |

Interpretation:

- current Mosaic `baseline`, `latency`, and `throughput` are structurally behind
- `throughput` does not reach crossover on the tested long-context shapes

### 2. `block_kv=64` remains the best measured default

Targeted and full-grid tuning summaries:

- [kernel_considerations_full_summary.json](./artifacts/kernel_considerations_full_summary.json)
- [bq64_wg1.json](./artifacts/bq64_wg1.json)
- [bq128_wg2_barrier_off.json](./artifacts/bq128_wg2_barrier_off.json)

Representative results:

- baseline tuning best: `block_q=64`, `block_kv=64`, `num_compute_wgs=1`, `132.86 ms`
- latency tuning best: `block_q=64`, `block_kv=64`, `num_compute_wgs=2`, `barrier=off`, `max_concurrent_steps=2`, `166.98 ms`
- throughput tuning best: `block_q=64`, `block_kv=64`, `num_compute_wgs=2`, `barrier=on`, `split_k=1`, `152.66 ms`

Interpretation:

- `block_kv=128` and `256` remain regressions on the tested shapes
- larger K/V tiles are not recovering enough overlap to justify the shared-memory cost

### 3. Barriers and deeper pipelines have real cost

Targeted artifacts:

- [steps2.json](./artifacts/steps2.json)
- [steps3.json](./artifacts/steps3.json)
- [steps4.json](./artifacts/steps4.json)
- [bq128_wg2_barrier_off.json](./artifacts/bq128_wg2_barrier_off.json)

Representative results:

- `block_q=128`, `num_compute_wgs=2`, barrier on: `157.99 ms`
- `block_q=128`, `num_compute_wgs=2`, barrier off: `147.33 ms`
- `max_concurrent_steps=2`: `156.35 ms`
- `max_concurrent_steps=3`: `153.61 ms`
- `max_concurrent_steps=4`: `163.26 ms`

Interpretation:

- barriers are not free
- pipeline depth is not monotonic
- the current kernel is sensitive to synchronization and shared-memory pressure

### 4. Split-k changes the current throughput bridge but does not fix it

Targeted split-k artifacts:

- [splitk_0.json](./artifacts/splitk_0.json)
- [splitk_1.json](./artifacts/splitk_1.json)
- [splitk_4.json](./artifacts/splitk_4.json)

Representative results at `batch=512`, `head_dim=128`, `max_blocks_per_seq=32`:

- heuristic / `split_k=0`: `157.09 ms`
- `split_k=1`: `161.25 ms`
- `split_k=4`: `176.24 ms`

Interpretation:

- split-k is a tuning knob on the current bridge
- it is not the missing ingredient that closes the gap to `blockwise`

### 5. The eager KV-update micro-attribution still favors `scatter`

KV artifacts:

- [kv_matrix_blockwise_eager.json](./artifacts/kv_matrix_blockwise_eager.json)
- [kv_matrix_blockwise_non_eager_fixed.json](./artifacts/kv_matrix_blockwise_non_eager_fixed.json)
- [blockwise_profile_summary.json](./artifacts/blockwise_profile_summary.json)

Representative eager results:

- `scatter`: `kv_update_total_s = 0.21894`, `2.35%` of model execute
- `compact_scatter`: `0.31168`, `7.08%`
- `sorted_compact_scatter`: `0.42182`, `9.33%`

All three ran the same work volume.

Interpretation:

- the current eager measurement does not justify a KV backend promotion away from `scatter`

## Where The Current Code Is Structurally Expensive

### Current throughput path

The current throughput path is still bridge-heavy:

- partition pages in the wrapper
- reshape partitioned block tables
- broadcast `q` across `k_splits`
- expand to a synthetic batch
- run the batched core
- reshape partials back
- merge partial results in JAX

Relevant code:

- [paged_decode_attention_mosaic_throughput()](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)
- [reduce_partitioned_decode_partials()](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)

### Current latency path

Latency still uses the same broad pattern with different partition sizing:

- [paged_decode_attention_mosaic_latency()](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)

### Current baseline batched core

The batched core looks metadata-heavy and synchronization-heavy:

- `prepare_decode_metadata(...)`
- `batched_decode_attention_mosaic(...)`

Relevant code:

- [prepare_decode_metadata()](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)
- [batched_decode_attention_mosaic()](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)

### Dispatch/runtime layer

The live runtime and dispatch boundary that owns current family selection and cache wiring:

- [_maybe_run_mosaic_decode()](../../../../src/nanovllm_jax/layers/paged_attention.py)
- [DecodeSchedulePacket](../../../../src/nanovllm_jax/engine/decode_schedule.py)
- [_prepare_decode()](../../../../src/nanovllm_jax/engine/model_runner.py)

## What Already Exists In This Branch For The Next Kernel

### 1. `throughput_v2` runtime seam

This branch adds a non-default `throughput_v2` family seam:

- [build_paged_decode_throughput_v2_plan()](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)
- [paged_decode_attention_mosaic_throughput_v2()](../../../../src/nanovllm_jax/layers/mosaic_gpu_attention.py)

Important current state:

- it is scaffolding only
- it still falls back to the current throughput implementation
- it exists to preserve a separate planning, cache, and dispatch boundary

### 2. Strict fresh-process synthetic A/B harness

This branch adds:

- [bench_decode_kernel_matrix.py](../../../../bench_decode_kernel_matrix.py)
- [decode_kernel_bench.py](../../../../src/nanovllm_jax/utils/decode_kernel_bench.py)

This exists so kernel claims can be gated by:

- fresh process per case
- explicit baseline comparison
- quick and stability passes
- machine-readable JSON outputs

### 3. Better decode runtime diagnostics

This branch also adds:

- decode-step timing summaries
- honest non-eager KV-measurement reporting
- schedule ownership diagnostics
- KV write counters

Relevant code:

- [runtime_diagnostics.py](../../../../src/nanovllm_jax/utils/runtime_diagnostics.py)
- [decode_profile_artifacts.py](../../../../src/nanovllm_jax/utils/decode_profile_artifacts.py)

## Current Hypothesis

Our current working hypothesis is:

1. do not keep broad-tuning the current throughput bridge
2. the next kernel should probably target the reduction/partition boundary explicitly
3. `throughput_v2` should own partitioning inside the kernel boundary rather than in wrapper-side `q` broadcast and reshape logic
4. schedule-owned metadata should feed that path directly

But this is still only a hypothesis. We want the expert to either confirm it or replace it with a better decomposition.

## Proposed Decision Space

Please explicitly choose among these, or propose a better one:

### Option A

Write a new long-context `throughput_v2` kernel that:

- keeps one logical query batch
- partitions K/V internally
- eliminates wrapper-side `q` broadcast
- keeps reduction out of Python/JAX wrapper code

### Option B

First write a dedicated partition-reduction kernel and keep the current partitioned compute kernel temporarily.

### Option C

Bypass the current batched Mosaic core entirely for the next decode kernel and write a different long-context decode operator with a new program decomposition.

### Option D

Do not pursue throughput first; instead replace the highest-overhead metadata or scheduling boundary before another kernel attempt.

## Benchmark Gate We Intend To Use

If the next kernel lands, we plan to validate it with:

1. Direct synthetic family bench against:
   - `blockwise`
   - current `throughput`
   - new candidate
2. Quick screen:
   - warmup `5`
   - iters `20`
3. Stability pass:
   - warmup `20`
   - iters `>= 200`
4. Real model decode profile after synthetic direction matches

Initial matrix:

- `batch=512`, `head_dim=128`, `max_blocks_per_seq=16`
- `batch=512`, `head_dim=128`, `max_blocks_per_seq=32`
- `batch=512`, `head_dim=128`, `max_blocks_per_seq=64`
- `batch=1024`, `head_dim=128`, `max_blocks_per_seq=16`
- `batch=2048`, `head_dim=128`, `max_blocks_per_seq=16`
- `batch=4096`, `head_dim=128`, `max_blocks_per_seq=16`

## Final Ask

Please recommend:

1. the next best step
2. the exact new kernel boundary
3. how to write that kernel in Pallas Mosaic GPU
4. which current code to keep, bypass, or replace
5. the minimum benchmark gate that should be required before we trust it
