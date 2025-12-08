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

# Pallas kernels (optional, depends on JAX version)
try:
    from .pallas_attention import (
        paged_attention,
        paged_decode_attention_kernel,
        PagedAttentionConfig,
        PALLAS_AVAILABLE,
    )
except ImportError:
    PALLAS_AVAILABLE = False

# True Mosaic GPU kernels (optional, requires Hopper+ GPU)
try:
    from .pallas_mosaic_attention import (
        MosaicAttentionConfig,
        batched_decode_attention_mosaic,
        prefill_attention_mosaic,
        paged_attention_mosaic,
        prefill_attention_mosaic_api,
        paged_decode_attention_mosaic_v2,
        simple_attention_wgmma,
        batched_decode_emit_pipeline,
        PagedKVBlockInfo,
        MOSAIC_AVAILABLE,
    )
except ImportError:
    MOSAIC_AVAILABLE = False
