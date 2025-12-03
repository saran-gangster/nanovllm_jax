"""Main LLM interface for nano-vllm JAX.

Provides a simple high-level API for text generation.
"""

from nanovllm_jax.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """High-level LLM interface for text generation.
    
    This is the main entry point for using nano-vllm JAX.
    
    Example:
        ```python
        from nanovllm_jax import LLM, SamplingParams
        
        llm = LLM("path/to/model")
        outputs = llm.generate(
            ["Hello, how are you?"],
            SamplingParams(temperature=0.7, max_tokens=100)
        )
        print(outputs[0]["text"])
        ```
    
    Args:
        model: Path to the HuggingFace model directory.
        max_num_batched_tokens: Maximum tokens in a batch (default: 16384).
        max_num_seqs: Maximum sequences in a batch (default: 512).
        max_model_len: Maximum sequence length (default: 4096).
        gpu_memory_utilization: Fraction of GPU memory for KV cache (default: 0.9).
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1).
        enforce_eager: Disable JIT compilation for debugging (default: False).
        kvcache_block_size: Tokens per KV cache block (default: 256).
    """
    pass
