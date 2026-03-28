# Nano-vLLM (JAX)

Minimal vLLM-style inference in **JAX/Flax NNX**, with paged KV-cache, prefix caching, and optional Pallas/Mosaic GPU decode kernels.

The implementation lives in `src/nanovllm_jax/`.

Current release focus:
- local single-node inference
- paged KV-cache execution
- public decode backends: `auto`, `blockwise`, `mosaic`
- supported runtime contract: `tensor_parallel_size=1`

## Project Status

The local JAX inference runtime is the supported focus of this repository.

The Mosaic GPU decode kernels are still an experimental preview. Early H100 bring-up results were promising, but further tuning and validation are temporarily paused pending renewed access to dedicated Hopper-class GPU hardware. Treat the Mosaic path as an under-development option rather than a production performance guarantee.

This repository is intentionally narrow. It is not presenting distributed serving, production tensor parallelism, or a finalized Mosaic kernel stack as complete product surfaces in this release.

## Install

Install JAX for your platform first, then install the package in editable mode:

```bash
pip install "jax[cpu]>=0.9.1"
pip install -e ".[dev]"
```

For CUDA/H100 environments, replace the CPU JAX wheel with the appropriate GPU JAX install for your platform.

## Model Download

Download model weights locally (example: Qwen3-0.6B):

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ./models/qwen/Qwen-3-0.6B \
  --local-dir-use-symlinks False
```

## Quick Start

Python API:

```python
from nanovllm_jax import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello"], sampling_params)
print(outputs[0]["text"])
```

Current runtime contract:
- `tensor_parallel_size=1` only
- public decode backends: `auto`, `mosaic`, `blockwise`
- internal Mosaic kernel families remain private to the runtime

Quick script:

```bash
python example_jax.py
```

## Architecture

The runtime has three main pieces:
- `LLM` / `LLMEngine`: request lifecycle, scheduling, and generation orchestration
- `ModelRunner`: model execution, KV-cache ownership, and decode/prefill preparation
- `src/nanovllm_jax/layers/`: model blocks plus paged attention dispatch and optional Mosaic GPU kernels

For decode attention, the public runtime chooses between:
- `blockwise`: portable streaming decode fallback
- `mosaic`: experimental Mosaic GPU decode path on Hopper-class GPUs
- `auto`: conservative runtime selection between the supported paths

The Mosaic implementation itself is split into a stable baseline kernel plus latency-focused and throughput-focused internal families. Those kernel-family names are implementation details, not part of the public config API.

More detail: [ARCHITECTURE.md](ARCHITECTURE.md)

## Benchmarks

- Quick: `python bench_jax.py`
