# nano-vLLM JAX

A compact JAX/Flax NNX inference runtime with paged KV cache support.

The project is intentionally small: it focuses on local single-node inference,
clear runtime boundaries, and a CPU-testable codebase. Experimental Hopper
Pallas Mosaic decode kernels are included behind runtime gates, but the stable
public surface is the JAX runtime itself.

## What Is Included

- `src/nanovllm_jax/engine`: request scheduling, batching, and model execution
- `src/nanovllm_jax/models`: model definitions and weight loading
- `src/nanovllm_jax/layers`: attention, MLP, normalization, embeddings, sampling
- `src/nanovllm_jax/utils`: runtime helpers, diagnostics, and profiling utilities
- `tests`: CPU-safe regression coverage for runtime behavior and dispatch logic

## Requirements

- Python `>=3.10,<3.13`
- JAX installed for your platform
- Local Hugging Face model weights

Install for CPU development:

```bash
python -m pip install -U pip
python -m pip install "jax[cpu]>=0.9.1"
python -m pip install -e ".[dev]"
```

For CUDA/H100 environments, install the matching GPU JAX wheel instead of the
CPU wheel.

## Quick Start

Download a local model first. For example:

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ./models/qwen/Qwen3-0.6B
```

Run a minimal generation:

```python
from nanovllm_jax import LLM, SamplingParams

llm = LLM("./models/qwen/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
params = SamplingParams(temperature=0.6, max_tokens=64)
outputs = llm.generate(["Write one sentence about JAX."], params)
print(outputs[0]["text"])
```

Or use the example script:

```bash
python example_jax.py
```

## Runtime Contract

- `tensor_parallel_size=1`
- local model directories only
- paged KV cache block size must be divisible by `256`
- decode backends are selected through the runtime config and environment gates
- Pallas Mosaic GPU kernels are experimental and intended for Hopper-class GPUs

## Development

Run the CPU-safe test suite:

```bash
python -m pytest tests -q
```

Run the smoke script:

```bash
python tests/smoke_jax.py
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current module map.
