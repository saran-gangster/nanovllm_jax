"""Neural network layers for LLM inference."""

from nanovllm_jax.layers.linear import (
    LinearBase,
    ReplicatedLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from nanovllm_jax.layers.attention import Attention, KVCache
from nanovllm_jax.layers.rotary_embedding import RotaryEmbedding, get_rope, apply_rotary_emb
from nanovllm_jax.layers.layernorm import RMSNorm
from nanovllm_jax.layers.activation import SiluAndMul
from nanovllm_jax.layers.sampler import Sampler
from nanovllm_jax.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

# Pallas kernels (optional, depends on JAX version)
try:
    from nanovllm_jax.layers.pallas_attention import (
        paged_attention,
        paged_decode_attention_kernel,
        PagedAttentionConfig,
        PALLAS_AVAILABLE,
    )
except ImportError:
    PALLAS_AVAILABLE = False

# True Mosaic GPU kernels (optional, requires Hopper+ GPU)
try:
    from nanovllm_jax.layers.pallas_mosaic_attention import (
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
