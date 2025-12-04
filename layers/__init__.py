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
