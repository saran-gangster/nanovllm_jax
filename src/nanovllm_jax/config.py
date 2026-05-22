import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """Configuration for the LLM engine.
    
    Attributes:
        model: Path to the HuggingFace model directory.
        max_num_batched_tokens: Maximum number of tokens in a batch.
        max_num_seqs: Maximum number of sequences in a batch.
        max_model_len: Maximum sequence length supported.
        gpu_memory_utilization: Fraction of GPU memory to use for KV cache.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        enforce_eager: If True, disable JIT compilation for debugging.
        hf_config: HuggingFace model configuration (auto-loaded).
        eos: End-of-sequence token ID (auto-loaded from config).
        kvcache_block_size: Number of tokens per KV cache block.
        num_kvcache_blocks: Number of KV cache blocks (-1 = auto-calculate).
    """
    model: str
    max_num_batched_tokens: int = 1024
    max_num_seqs: int = 16
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Any = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Decode batching: pad batch size to reduce recompilations / improve GEMM efficiency.
    # (Will be clamped to max_num_seqs.)
    # Default to no minimum padding (1). Users can increase this (e.g. 8/16)
    # to trade extra compute for fewer compiled shapes / potentially better
    # small-batch GEMM efficiency on their hardware.
    decode_batch_min_size: int = 1

    # Attention backend selection (overrides env vars when not "auto").
    # Decode: auto|mosaic|blockwise
    decode_attention_backend: str = "auto"
    # Prefill: auto|cudnn|xla|mosaic
    prefill_attention_backend: str = "auto"

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"Model path does not exist: {self.model}")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        if self.decode_batch_min_size < 1:
            raise ValueError("decode_batch_min_size must be >= 1")
        if self.gpu_memory_utilization <= 0.0:
            raise ValueError("gpu_memory_utilization must be > 0")
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be divisible by 256")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0:
            raise ValueError("num_kvcache_blocks must be -1 or > 0")
        if self.tensor_parallel_size != 1:
            raise ValueError(
                "tensor_parallel_size=1 is the only supported mode in this release."
            )

        decode_backend = str(self.decode_attention_backend).strip().lower()
        if decode_backend not in {"auto", "mosaic", "blockwise"}:
            raise ValueError(
                "decode_attention_backend must be one of: auto|mosaic|blockwise"
            )
        self.decode_attention_backend = decode_backend

        prefill_backend = str(self.prefill_attention_backend).strip().lower()
        if prefill_backend not in {"auto", "cudnn", "xla", "mosaic"}:
            raise ValueError(
                "prefill_attention_backend must be one of: auto|cudnn|xla|mosaic"
            )
        self.prefill_attention_backend = prefill_backend

        # Lazy import: keeps host-side scheduler/unit benchmarks lightweight and
        # avoids importing optional heavyweight dependencies until needed.
        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len")
