# Kernel Development Considerations for `nano-vllm` on H100

Status: this note is the current H100 kernel-development guide for the live codebase. It is based on direct GPU measurements collected on 2026-04-12 and is meant to drive better kernel design, narrower tuning sweeps, and cleaner promotion decisions.

## Goal

The goal is not to restate that `blockwise` is faster. The goal is to identify:

- which knobs actually change kernel topology
- which knobs mostly trade shared memory against overlap
- where the current Mosaic decode families are structurally expensive
- what should be measured first before writing or promoting a new kernel

This document should be read alongside:

- [benchmarks/results_h100_2026-04-12.md](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/results_h100_2026-04-12.md)
- [good_bowl/throughput_v2_checklist.md](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/good_bowl/throughput_v2_checklist.md)
- [bench_decode_families.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/bench_decode_families.py)
- [src/nanovllm_jax/layers/mosaic_gpu_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/mosaic_gpu_attention.py)
- [src/nanovllm_jax/layers/paged_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/paged_attention.py)

Raw artifacts used here:

- [benchmarks/h100_2026-04-12/family_sweep](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/family_sweep)
- [benchmarks/h100_2026-04-12/extended_sweeps](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/extended_sweeps)
- [benchmarks/h100_2026-04-12/kernel_considerations](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kernel_considerations)
- [benchmarks/h100_2026-04-12/kernel_considerations_splitk](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kernel_considerations_splitk)
- [benchmarks/h100_2026-04-12/kernel_considerations_full](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kernel_considerations_full)
- [benchmarks/h100_2026-04-12/kv_matrix_blockwise_eager](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kv_matrix_blockwise_eager)
- [benchmarks/h100_2026-04-12/kv_matrix_blockwise_non_eager_fixed](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kv_matrix_blockwise_non_eager_fixed)

## H100 Environment

The GPU machine used for the measurements:

- GPU: `NVIDIA H100 80GB HBM3`
- Driver: `580.126.09`
- Python: `3.11.10`
- JAX: `0.9.2`
- JAX backend: `gpu`
- JAX device: `CudaDevice(id=0)`

One live `nvidia-smi` sample taken while tuning was running:

- P-state: `P0`
- graphics clock: `1980 MHz`
- SM clock: `1980 MHz`
- memory clock: `2619 MHz`
- power draw: `164.90 W / 700.00 W`
- temperature: `38 C`
- GPU utilization: `26 %`
- memory use: `61373 MiB / 81559 MiB`

Interpretation:

- the memory footprint is meaningful
- the point-sampled utilization and power draw are not steady-state kernel truth and should not be overinterpreted
- this is a single-GPU box with no NVLink path to another GPU, so all data here is single-device kernel behavior

The topology snapshot is still useful operationally:

- GPU NUMA affinity: `0`
- closest NIC path includes `PIX` for `mlx5_3` and `mlx5_4`
- most other NIC paths are `NODE` or `SYS`

This matters mainly for future networking or multi-process experiments, not for the current single-GPU decode kernel work.

## Measurement Provenance

The data in this note comes from four classes of runs.

### 1. Direct family sweeps

Script:

- [bench_decode_families.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/bench_decode_families.py)

Purpose:

- compare `blockwise`, `baseline`, `latency`, and `throughput` under the same synthetic decode shapes

Artifacts:

- [benchmarks/h100_2026-04-12/family_sweep/summary.json](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/family_sweep/summary.json)
- [benchmarks/h100_2026-04-12/extended_sweeps/summary.json](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/extended_sweeps/summary.json)

### 2. Targeted topology and pipeline sweeps

Purpose:

- isolate the knobs that actually change decode topology and pipeline depth:
  - `block_q`
  - `block_kv`
  - `num_compute_wgs`
  - `use_schedule_barrier`
  - `max_concurrent_steps`
  - `throughput_split_k`

Artifacts:

- [benchmarks/h100_2026-04-12/kernel_considerations](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kernel_considerations)
- [benchmarks/h100_2026-04-12/kernel_considerations_splitk](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kernel_considerations_splitk)
- [benchmarks/h100_2026-04-12/kernel_considerations_full/summary.json](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kernel_considerations_full/summary.json)

### 3. Real model decode runtime profiles

Scripts:

- [profile_decode_runtime.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/profile_decode_runtime.py)
- [profile_kv_update_backends.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/profile_kv_update_backends.py)

Purpose:

- confirm end-to-end decode behavior
- validate KV-update backend attribution
- separate eager-measurable subcomponent timings from non-eager/JIT paths where subcomponent timing is unavailable

Artifacts:

- [benchmarks/h100_2026-04-12/blockwise_smoke](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/blockwise_smoke)
- [benchmarks/h100_2026-04-12/blockwise_profile](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/blockwise_profile)
- [benchmarks/h100_2026-04-12/kv_matrix_blockwise_eager](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kv_matrix_blockwise_eager)
- [benchmarks/h100_2026-04-12/kv_matrix_blockwise_non_eager_fixed](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kv_matrix_blockwise_non_eager_fixed)

### 4. Unit and regression checks on the GPU box

Purpose:

- validate the instrumentation and schedule/cache ownership changes on the same machine before trusting the benchmark outputs

Result:

- targeted non-model suites passed on the GPU machine

## Timing Caveats

Three timing caveats matter for future kernel work.

### Eager vs non-eager subcomponent timing

In eager mode, the runtime can attribute `kv_update_s` directly enough to compare KV backends honestly.

In non-eager/JIT mode:

- backend attribution is still correct
- `kv_update_s` is not currently measurable through the same path
- summaries therefore report configured backend plus `kv_update_measured=False`

Implication:

- use eager mode for KV micro-attribution
- use non-eager runs for end-to-end behavior, but do not pretend the missing subcomponent timing exists

### Compile-and-first-run is not steady-state

Every artifact here records compile-and-first-run separately from iterative timings. Promotion decisions should use steady-state `mean`, `p50`, and `min`, not compile time alone.

Compile time is still operationally important because it affects:

- tuning iteration speed
- sensitivity to recompilation
- the runtime cost of shape churn

### CUDA timer warnings under non-eager runs

Some non-eager runtime profiling emitted delay-kernel timer warnings from JAX. The runs still completed and the top-level results were usable, but that is another reason not to overclaim fine-grained subcomponent timing from non-eager traces.

## What The Code Makes Expensive

The most important thing the measurements say is not just that Mosaic is slower. It is where the current code structure is making that likely.

### `blockwise` has the simple path

The blockwise decode path in [paged_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/paged_attention.py) is comparatively direct:

- stream pages
- keep the online-softmax style reduction local to the decode operator
- avoid split-batch bridge plumbing

That simplicity shows up in both steady-state time and compile cost.

### The Mosaic path pays extra before the kernel starts

The Mosaic dispatch path in [paged_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/paged_attention.py) still does extra work that `blockwise` does not:

- `_maybe_run_mosaic_decode()`
- `_ensure_mosaic_decode_probe_ready()`
- variant/probe/fallback handling

That is not necessarily the dominant steady-state cost, but it does mean Mosaic starts from a more complicated runtime boundary.

### `batched_decode_attention_mosaic()` is metadata-heavy

The batched Mosaic decode implementation in [mosaic_gpu_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/mosaic_gpu_attention.py) materializes and reshapes a large amount of tile metadata:

- `block_tables.reshape(...)`
- `context_lens.reshape(...)`
- repeated `broadcast_to(...)`
- flattened chunk-position and row-index tables

That work is not free. The benchmark artifacts record baseline metadata preparation around `1.74 s` for the targeted topology sweep shape before the iter loop. That is not per-step runtime, but it is real setup work and it indicates the path is structurally metadata-intensive.

Two concrete metadata-shape examples from the targeted baseline artifacts:

- `block_q=64`:
  - `tile_chunk_counts=[8]`
  - `tile_chunk_row_indices=[8, 8192]`
- `block_q=128`:
  - `tile_chunk_counts=[4]`
  - `tile_chunk_row_indices=[4, 16384]`

Interpretation:

- increasing `block_q` changes the batching topology, not just the tile size
- the metadata packing shifts substantially with `block_q`
- a kernel change that looks like “just use a bigger `block_q`” is actually a topology change with different metadata behavior and resource tradeoffs

### `latency` and `throughput` are still bridge designs

The current Mosaic latency and throughput decode families in [mosaic_gpu_attention.py](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/src/nanovllm_jax/layers/mosaic_gpu_attention.py) still wrap the batched core with extra bridge logic:

- split pages into partitions
- reshape partitioned block tables
- broadcast `q` across `k_splits`
- reshape into a larger synthetic batch
- run the batched core
- reshape partial outputs back
- merge per-partition accumulators in JAX

That is exactly the structure that shows up around:

- `paged_decode_attention_mosaic_latency(...)`
- `paged_decode_attention_mosaic_throughput(...)`

This is the core architectural reason the current throughput family is not competitive. `split_k` changes this bridge behavior, but it does not remove the bridge.

## High-Level H100 Conclusions

The current H100 data supports six concrete conclusions.

1. `blockwise` is still the best decode family on every tested shape.
2. Current Mosaic `baseline`, `latency`, and `throughput` are not close enough to justify more broad retuning before structural cleanup.
3. `block_kv=64` is the strongest measured default so far.
4. multi-WG tuning only becomes a reliable optimization path once `block_q` is large enough to make the live decode topology credible.
5. `use_schedule_barrier` has measurable cost and must be justified by a win, not assumed to help.
6. `throughput_split_k` changes behavior, but it does not rescue the current throughput bridge.

## Family-Level Results

### Same-shape batch sweep

Common shape:

- `head_dim=128`
- `max_blocks_per_seq=16`
- `block_size=256`
- `num_heads=16`
- `num_kv_heads=8`

Measured mean latencies:

| Batch | Blockwise | Baseline | Latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| `64` | `1.56 ms` | `135.58 ms` | `147.35 ms` | `151.99 ms` |
| `512` | `9.45 ms` | `140.94 ms` | `144.41 ms` | `150.94 ms` |
| `2048` | `36.35 ms` | `149.79 ms` | `165.57 ms` | `167.30 ms` |
| `4096` | `72.15 ms` | `172.42 ms` | `181.77 ms` | `185.39 ms` |

Speed ratios against `blockwise`:

- batch `64`
  - `baseline`: about `87.1x` slower
  - `latency`: about `94.7x` slower
  - `throughput`: about `97.7x` slower
- batch `512`
  - `baseline`: about `14.9x` slower
  - `latency`: about `15.3x` slower
  - `throughput`: about `16.0x` slower
- batch `4096`
  - `baseline`: about `2.4x` slower
  - `latency`: about `2.5x` slower
  - `throughput`: about `2.6x` slower

Interpretation:

- Mosaic becomes less catastrophically behind as batch grows, but it never catches up on the tested shapes
- the throughput path does not show a long-context crossover in these measurements

### Long-context fixed-batch sweep

Fixed batch:

- `batch=512`
- `head_dim=128`

Measured mean latencies:

| Blocks / Seq | Blockwise | Baseline | Latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| `32` | `18.65 ms` | `133.84 ms` | `156.15 ms` | `155.15 ms` |
| `64` | `37.10 ms` | `147.41 ms` | `169.88 ms` | `150.32 ms` |

Interpretation:

- longer context does not currently produce a throughput-family win
- at `64` blocks per sequence, throughput does beat latency, but it is still far behind `blockwise`

### Head-dim sensitivity

Shape:

- `batch=512`
- `max_blocks_per_seq=16`

Measured mean latencies at `head_dim=64`:

| Family | Mean |
| --- | ---: |
| `blockwise` | `5.34 ms` |
| `baseline` | `123.62 ms` |
| `latency` | `130.08 ms` |
| `throughput` | `133.65 ms` |

Interpretation:

- reducing head dim helps all families
- it does not change the ranking
- current Mosaic losses are not explained by `head_dim=128` alone

## Broader Cross-Product Sweep Confirmation

The larger H100 sweep in [benchmarks/h100_2026-04-12/kernel_considerations_full/summary.json](/home/saran-gangster/Desktop/Implementations%20&%20Projects/jax/nano-vllm/benchmarks/h100_2026-04-12/kernel_considerations_full/summary.json) tested `68` configurations across three focused grids:

- baseline tuning at `batch=512`, `max_blocks_per_seq=16`
- latency tuning at `batch=512`, `max_blocks_per_seq=64`
- throughput tuning at `batch=512`, `max_blocks_per_seq=64`

Best result per sweep:

| Sweep | Best Config | Mean | Compile |
| --- | --- | ---: | ---: |
| `baseline_tuning` | `block_q=64`, `block_kv=64`, `num_compute_wgs=1` | `132.86 ms` | `0.65 s` |
| `latency_tuning` | `block_q=64`, `block_kv=64`, `num_compute_wgs=2`, `barrier=off`, `max_concurrent_steps=2` | `166.98 ms` | `3.20 s` |
| `throughput_tuning` | `block_q=64`, `block_kv=64`, `num_compute_wgs=2`, `barrier=on`, `split_k=1` | `152.66 ms` | `2.39 s` |

What the broader grid confirms:

- `block_kv=64` remains the best value across all three grids
- `block_kv=128` and especially `256` remain regressions
- shallow pipelines still beat deeper ones on the tested latency shapes
- throughput can move within its own family, but its best measured point is still far behind `blockwise`

One important caveat:

- the synthetic tuning harness can label configurations that do not map 1:1 onto live decode selection behavior
- live decode still has its own topology constraints and guardrails
- use the full grid for kernel search direction, but validate any promotion candidate through the live decode path before treating it as a real runtime default

## Compile Cost Matters More Than It Looks

The family sweeps also show a clear compile-first-run pattern:

- `blockwise`: roughly `1.0 s` to `1.5 s`
- `baseline`: roughly `0.5 s` to `0.75 s`
- `latency`: roughly `2.2 s` to `3.3 s`
- `throughput`: roughly `2.4 s` to `3.5 s`

Interpretation:

- latency and throughput are not only slower at steady state
- they are also heavier to iterate on
- that matters for tuning velocity and for any runtime path that can trigger recompilation from shape churn

This is another argument for simplifying the throughput path instead of treating it as a tuning-only problem.

## Knob-by-Knob Considerations

## `block_q`

`block_q` is not just a tile size. It is a topology gate.

In live decode, `_select_decode_num_compute_wgs()` constrains when multi-WG execution is meaningful. The practical effect is:

- `block_q=64` strongly biases the live path toward a simpler single-WG-style topology
- `block_q=128` is the first value that clearly opens up a more credible 2-WG decode topology
- synthetic tuning runs can still label or force additional combinations, but those must be revalidated against live decode selection behavior before promotion

That means `block_q` controls:

- rows per compute warpgroup
- whether warp specialization is even available
- metadata packing shape
- buffer/barrier pressure for the rest of the kernel

Measured baseline topology sweep at:

- `batch=512`
- `max_blocks_per_seq=32`
- `head_dim=128`
- `block_kv=64`
- `max_concurrent_steps=2`

| Config | Mean |
| --- | ---: |
| `block_q=64`, `num_compute_wgs=1` | `137.18 ms` |
| `block_q=128`, `num_compute_wgs=1` | `224.33 ms` |
| `block_q=128`, `num_compute_wgs=2`, `barrier=on` | `157.99 ms` |
| `block_q=128`, `num_compute_wgs=2`, `barrier=off` | `147.33 ms` |

Interpretation:

- `block_q=128` is not a default win
- the only reason to use it today is to unlock `num_compute_wgs=2`
- even then, the best measured `block_q=128` result still lost to `block_q=64`

Practical rule:

- do not treat `num_compute_wgs` as a first-class tuning knob until `block_q` makes the intended topology credible in the live path

## `num_compute_wgs`

This knob matters only after the `block_q` gate is crossed.

Measured effect at `block_q=128`:

- `1 WG`: `224.33 ms`
- `2 WG` with barrier on: `157.99 ms`
- `2 WG` with barrier off: `147.33 ms`

Interpretation:

- 2-WG execution can recover a large part of the `block_q=128` regression
- but it does not automatically beat the simpler `block_q=64` path
- multi-WG decode currently looks like a conditional optimization, not the universal answer

Kernel-design implication:

- when testing a new decode kernel, first decide whether the kernel is fundamentally designed for one WG or multiple WGs
- do not mix that design question with broad random sweeps

## `use_schedule_barrier`

This is a coordination knob with measurable cost.

At the targeted `block_q=128`, `num_compute_wgs=2` shape:

- barrier on: `157.99 ms`
- barrier off: `147.33 ms`

Delta:

- about `10.66 ms`
- about `6.8 %` of the barrier-on runtime

Interpretation:

- barriers are expensive enough to matter
- they must be justified by improved overlap, correctness, or robustness
- they should not be treated as a safe default

Practical rule:

- if a new kernel needs schedule barriers, prove that they buy more than they cost

## `block_kv`

`block_kv` is the most important steady-state tile-size knob today.

It directly controls:

- K/V pipeline stage size
- chunks per block
- pipeline depth budget
- per-stage shared memory footprint

Measured examples:

Baseline family:

| Shape | `block_kv=64` | `block_kv=128` | `block_kv=256` |
| --- | ---: | ---: | ---: |
| `batch=512`, `blocks/seq=16`, `head_dim=128` | `120.16 ms` | `130.93 ms` | `171.92 ms` |

More shapes from earlier H100 notes:

- `batch=768`, `blocks/seq=32`, `head_dim=128`, baseline:
  - `64`: `138.67 ms`
  - `256`: `184.62 ms`
- `batch=512`, `blocks/seq=48`, `head_dim=128`, throughput:
  - `64`: `139.28 ms`
  - `256`: `181.90 ms`

Interpretation:

- `block_kv=64` is the safest measured default so far
- larger `block_kv` values appear to inflate stage cost faster than they recover useful overlap
- `256` should be treated as a suspect value until a shape proves otherwise

Practical rule:

- use `64` as the baseline comparison point for any new decode kernel

## `max_concurrent_steps`

This is the K/V pipeline depth knob.

It changes:

- number of in-flight K/V stages
- staging-buffer usage
- barrier count
- occupancy pressure through shared memory

Measured at:

- `block_q=128`
- `block_kv=64`
- `num_compute_wgs=2`
- `use_schedule_barrier=false`
- `batch=512`
- `max_blocks_per_seq=32`

| `max_concurrent_steps` | Mean |
| --- | ---: |
| `2` | `156.35 ms` |
| `3` | `153.61 ms` |
| `4` | `163.26 ms` |

Interpretation:

- the optimum is local, not monotonic
- `3` beat `2` slightly
- `4` regressed

This matches the code-level expectation:

- deeper pipelines improve potential overlap
- deeper pipelines also increase shared-memory pressure and synchronization cost
- once that cost exceeds the overlap benefit, performance reverses

Practical rule:

- always cap this with measurement
- do not assume deeper is better

## `throughput_split_k`

This knob only tunes the current throughput bridge. It does not replace it.

Measured at:

- family: `throughput`
- `batch=512`
- `max_blocks_per_seq=32`
- `head_dim=128`
- `block_q=128`
- `block_kv=64`
- `num_compute_wgs=2`
- `use_schedule_barrier=true`
- `max_concurrent_steps=2`

| `split_k` | Expected Split Count | Mean |
| --- | --- | ---: |
| `0` | heuristic, measured `8` | `157.09 ms` |
| `1` | `1` | `161.25 ms` |
| `2` | `2` | `164.58 ms` |
| `4` | `4` | `176.24 ms` |
| `8` | `8` | `161.09 ms` |

Interpretation:

- the heuristic path was best on this shape
- higher split counts were not a reliable win
- even the best split count stayed far behind `blockwise`

Practical rule:

- if split-k tuning is all that changes, do not expect a breakthrough
- a future `throughput-v2` must shrink or remove the split/broadcast/merge bridge

## KV-Update Backend Findings

Kernel development here is not only about attention. The KV write path matters too.

Eager real-model matrix on H100:

- `scatter`:
  - `kv_update_total_s = 0.21894`
  - `share_of_model_execute = 2.35 %`
- `compact_scatter`:
  - `0.31168`
  - `7.08 %`
- `sorted_compact_scatter`:
  - `0.42182`
  - `9.33 %`

All three backends saw the same work:

- `kv_update_calls = 868`
- `kv_update_tokens = 1736`
- `kv_update_valid_tokens = 1736`
- `kv_update_skipped_tokens = 0`
- `kv_update_duplicate_slots = 0`

Interpretation:

- on the valid eager micro-attribution path, `scatter` is the best measured backend
- compact variants were slower on the same write volume

Important caveat:

- in non-eager/JIT mode, backend attribution is now correct, but `kv_update_s` is not measurable through the current timing path
- the non-eager summaries therefore report configured backend plus `kv_update_measured=False`

Practical rule:

- do not promote a KV backend on non-eager wall-clock alone
- use eager micro-attribution or build a dedicated microbenchmark path first

## What To Tune First

If we need the smallest useful H100 tuning order for a new decode kernel, it should be:

1. topology gate:
   - `block_q`
   - `num_compute_wgs`
   - barrier on or off for the multi-WG case
2. `block_kv`
3. `max_concurrent_steps`
4. `throughput_split_k`
5. only then any finer geometry retuning

Why this order is correct:

- `block_q` determines whether the multi-WG topology exists at all
- `block_kv` changes the pipeline budget
- `max_concurrent_steps` is only meaningful after tile geometry is stable
- `split_k` is a late-stage bridge knob, not a first-principles topology knob

## What To Avoid

These are low-value paths based on the current data.

- broad random sweep matrices without first resolving the topology gate
- treating `block_q=128` as a safe default
- treating schedule barriers as harmless
- spending time on `max_concurrent_steps` before `block_q` and `block_kv`
- expecting `split_k` to fix the current throughput architecture
- promoting Mosaic variants based on one improved local shape without comparison to `blockwise`

## What A Better Kernel Should Change

The current measurements point to structural requirements for any serious replacement kernel.

### A better baseline-style decode kernel should

- minimize host-side or JAX-side metadata packing
- keep tile metadata compact and reusable across steps
- avoid topology choices that require expensive barriers unless they clearly improve overlap
- preserve a simple reduction boundary

### A better throughput kernel should

- avoid wrapper-level `q` broadcast across `k_splits`
- avoid synthetic batch expansion followed by reshape/merge
- move more of the partition reduction into the kernel boundary
- reduce or eliminate JAX-side partial-output merging

### A better KV-update path should

- keep `update_kv_cache()` as the public internal boundary
- replace the backend with a dedicated store kernel only after a microbenchmark proves it beats `scatter`
- measure write volume and slot behavior on every run so the comparison stays honest

## What To Record For Every New Kernel

For each new kernel or kernel family, record:

- exact git SHA
- exact command
- GPU model and driver
- JAX and Python versions
- shape:
  - batch
  - head_dim
  - max_blocks_per_seq
  - block size
  - number of heads
  - number of KV heads
- tuning config:
  - `block_q`
  - `block_kv`
  - `num_compute_wgs`
  - `use_schedule_barrier`
  - `max_concurrent_steps`
  - `throughput_split_k`
- compile-and-first-run time
- steady-state `min`, `p50`, and `mean`
- output checksum
- any metadata-preparation timing or cache state
- any eager-only subcomponent timing limitations

If a kernel claim matters, do two passes:

- quick screen:
  - low warmup
  - around 10 to 20 iterations
- stability check:
  - higher warmup
  - around 200 or more iterations

If the quick screen and long run disagree, do not report the kernel as improved.

## Minimal Command Set

These are the commands that matter most for future work.

Direct family comparison:

```bash
cd /workspace/nanovllm_jax && \
PYTHONPATH=src \
python3 bench_decode_families.py \
  --family blockwise \
  --batch-size 512 \
  --num-heads 16 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-size 256 \
  --max-blocks-per-seq 32 \
  --num-blocks 4096 \
  --dtype bfloat16 \
  --warmup 5 \
  --iters 20
```

Targeted topology sweep:

```bash
cd /workspace/nanovllm_jax && \
PYTHONPATH=src \
python3 bench_decode_families.py \
  --family baseline \
  --batch-size 512 \
  --num-heads 16 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-size 256 \
  --max-blocks-per-seq 32 \
  --num-blocks 4096 \
  --dtype bfloat16 \
  --block-kv 64 \
  --block-q 128 \
  --num-compute-wgs 2 \
  --use-schedule-barrier false \
  --max-concurrent-steps 2 \
  --warmup 5 \
  --iters 10
```

Throughput split-k sweep:

```bash
cd /workspace/nanovllm_jax && \
PYTHONPATH=src \
python3 bench_decode_families.py \
  --family throughput \
  --batch-size 512 \
  --num-heads 16 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-size 256 \
  --max-blocks-per-seq 32 \
  --num-blocks 4096 \
  --dtype bfloat16 \
  --block-kv 64 \
  --block-q 128 \
  --num-compute-wgs 2 \
  --use-schedule-barrier true \
  --max-concurrent-steps 2 \
  --throughput-split-k 0 \
  --warmup 5 \
  --iters 10
```

Real-model KV backend attribution:

```bash
cd /workspace/nanovllm_jax && \
PYTHONPATH=src \
NANOVLLM_MODEL_PATH=/workspace/models/Qwen3-0.6B \
python3 profile_kv_update_backends.py \
  --output-dir /workspace/out/kv_matrix \
  --decode-backend blockwise \
  --backends scatter,compact_scatter,sorted_compact_scatter \
  --baseline-backend scatter \
  --max-tokens 32 \
  --temperature 0.0001 \
  --enforce-eager
```

## Bottom Line

The H100 measurements say the current bottleneck is mostly architectural, not just parametric.

- `blockwise` is simpler and still wins clearly
- current Mosaic paths pay for metadata shaping, split/broadcast/merge structure, and extra coordination
- `block_q`, `block_kv`, `num_compute_wgs`, `use_schedule_barrier`, `max_concurrent_steps`, and `split_k` all matter, but none of them alone closes the structural gap

The shortest correct path forward is:

1. use the current H100 data to stop broad unstructured tuning
2. keep `block_kv=64` as the default comparison point
3. treat `block_q` as a topology decision, not a harmless tile-size tweak
4. force every barrier and every extra pipeline stage to justify itself
5. build `throughput-v2` as a cleaner kernel boundary instead of continuing to tune the current bridge as though it were nearly done
