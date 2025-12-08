"""Setup script for nanovllm_jax package."""

from setuptools import setup, find_packages

setup(
    name="nanovllm_jax",
    version="0.1.0",
    description="A minimal JAX implementation of vLLM with Pallas Mosaic GPU kernels",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "flax>=0.8.0",
        "transformers>=4.51.0",
        "safetensors",
        "xxhash",
    ],
)
