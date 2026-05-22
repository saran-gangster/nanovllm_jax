"""Neural network layers for LLM inference."""

from .linear import (
    LinearBase,
    ReplicatedLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from .attention import Attention, KVCache
from .rotary_embedding import RotaryEmbedding, get_rope, apply_rotary_emb
from .layernorm import RMSNorm
from .activation import SiluAndMul
from .sampler import Sampler
from .embed_head import VocabParallelEmbedding, ParallelLMHead

# Paged attention kernels (optional, depends on JAX version)
try:
    from .paged_attention import (
        paged_attention as paged_attention_dispatch,
        paged_decode_attention_kernel,
        PagedAttentionConfig,
        PALLAS_AVAILABLE,
    )
except ImportError:
    PALLAS_AVAILABLE = False

# Mosaic GPU decode kernels (optional, requires Hopper+ GPU)
try:
    from .mosaic_gpu_attention import (
        MosaicAttentionConfig,
        batched_decode_attention_mosaic,
        paged_decode_attention_mosaic_latency,
        paged_decode_attention_mosaic_throughput,
        prefill_attention_mosaic,
        prefill_attention_mosaic_api,
        MOSAIC_AVAILABLE,
    )
except ImportError:
    MOSAIC_AVAILABLE = False
