"""Main LLM interface for nano-vllm JAX.

Provides a simple high-level API for text generation.
"""

from __future__ import annotations

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
        max_num_batched_tokens: Maximum tokens in a batch (default: 1024).
        max_num_seqs: Maximum sequences in a batch (default: 16).
        max_model_len: Maximum sequence length (default: 512).
        gpu_memory_utilization: Fraction of GPU memory for KV cache (default: 0.5).
        tensor_parallel_size: Supported value is 1.
        enforce_eager: Disable JIT compilation for debugging (default: False).
        kvcache_block_size: Tokens per KV cache block (default: 256).
    """
    def __init__(
        self,
        model: str,
        *,
        max_num_batched_tokens: int = 1024,
        max_num_seqs: int = 16,
        max_model_len: int = 512,
        gpu_memory_utilization: float = 0.5,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        kvcache_block_size: int = 256,
        num_kvcache_blocks: int = -1,
        decode_batch_min_size: int = 1,
        decode_attention_backend: str = "auto",
        prefill_attention_backend: str = "auto",
    ):
        super().__init__(
            model=model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            kvcache_block_size=kvcache_block_size,
            num_kvcache_blocks=num_kvcache_blocks,
            decode_batch_min_size=decode_batch_min_size,
            decode_attention_backend=decode_attention_backend,
            prefill_attention_backend=prefill_attention_backend,
        )
