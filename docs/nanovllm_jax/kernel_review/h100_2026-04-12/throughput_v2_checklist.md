# Throughput-v2 Checklist

This checklist is constrained by the live H100 data collected on 2026-04-12:

- current `blockwise` wins decisively across batch `64` through `4096`
- current Mosaic `throughput` is still slower than `blockwise` even at larger batch and longer context
- current `throughput` loss is dominated by structure, not by one obvious tile choice

## Goal

Build a real long-context decode operator that removes the current bridge overhead instead of tuning the existing split/broadcast/merge wrapper.

## Non-Goals

- do not retune the current `throughput` bridge as the main path
- do not promote current `latency` or current `baseline`
- do not couple throughput-v2 bring-up to KV-backend promotion

## Required Structural Changes

1. Add a new operator path beside the current throughput implementation.
   - Keep the current throughput path intact for comparison and fallback.
   - Do not mutate the existing bridge into place.

2. Remove wrapper-level `q` broadcast across partitioned splits.
   - Current bridge expands query work before the kernel even starts.
   - Throughput-v2 should keep one logical query batch and partition KV work internally.

3. Remove wrapper-level split reshaping of `block_tables` and `context_lens`.
   - The scheduling information should enter the operator in runner-owned schedule form.
   - Partitioning should be kernel-native or preparation-native, not Python/JAX wrapper plumbing.

4. Remove JAX-side partial-result merge from the main throughput path.
   - The current reduction boundary is a major structural cost.
   - Replace it with either:
     - in-kernel reduction, or
     - a dedicated reduction kernel boundary with explicit measurement.

5. Make decode schedule metadata the primary throughput-v2 input seam.
   - Use `DecodeSchedulePacket` data directly rather than rebuilding family-specific partition structure ad hoc.
   - Keep family metadata cache ownership explicit and per-schedule.

## File Targets

Primary implementation files:

- `src/nanovllm_jax/layers/mosaic_gpu_attention.py`
- `src/nanovllm_jax/layers/paged_attention.py`
- `src/nanovllm_jax/engine/decode_schedule.py`
- `src/nanovllm_jax/engine/model_runner.py`

Primary tests:

- `tests/test_mosaic_decode_family.py`
- `tests/test_mosaic_throughput_utils.py`
- `tests/test_decode_runtime_ownership.py`
- `tests/test_decode_profile_artifacts.py`

## Bring-Up Order

1. Define a separate `throughput_v2` family entry point.
2. Thread schedule-owned metadata into that path without changing runtime defaults.
3. Stand up a kernel path that avoids wrapper-side split/broadcast work.
4. Add an explicit reduction boundary if fully fused reduction is not ready.
5. Add direct synthetic benchmarking in `bench_decode_families.py`.
6. Add controlled runtime profiling through `profile_decode_runtime.py`.
7. Compare against:
   - current `blockwise`
   - current `throughput`
   - current `baseline`

## Validation Gates

Do not call throughput-v2 a win unless all of these are true:

1. Correctness against `blockwise` reference on at least one representative shape family.
2. Direct synthetic family bench shows throughput-v2 materially beats current throughput.
3. Direct synthetic family bench narrows the gap to `blockwise` or wins on at least one intended long-context region.
4. Controlled runtime profile shows the same direction on the real model path.
5. Quick-screen and longer confirmation runs agree.

## Initial Benchmark Matrix

Use these first:

1. `batch=512`, `head_dim=128`, `max_blocks_per_seq=16`
2. `batch=512`, `head_dim=128`, `max_blocks_per_seq=32`
3. `batch=512`, `head_dim=128`, `max_blocks_per_seq=64`
4. `batch=1024`, `head_dim=128`, `max_blocks_per_seq=16`
5. `batch=2048`, `head_dim=128`, `max_blocks_per_seq=16`
6. `batch=4096`, `head_dim=128`, `max_blocks_per_seq=16`

Optional follow-up:

1. `batch=512`, `head_dim=64`, `max_blocks_per_seq=16`

## Promotion Rule

If throughput-v2 does not beat the current bridge path quickly on direct synthetic measurement, stop tuning it and fix structure first.
