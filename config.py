import os
from dataclasses import dataclass
from transformers import AutoConfig


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
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model), f"Model path does not exist: {self.model}"
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size must be divisible by 256"
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size must be between 1 and 8"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len, \
            "max_num_batched_tokens must be >= max_model_len"
