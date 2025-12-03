"""Qwen3 model implementation in JAX with Flax NNX.

This module provides the Qwen3 transformer model architecture with:
- Tensor parallelism support via Mesh + NamedSharding + shard_map
- Paged attention with KV-cache
- RoPE positional embeddings
- GQA (Grouped Query Attention)
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
from transformers import Qwen2Config as Qwen3Config  # Qwen3 uses Qwen2Config

from nanovllm_jax.layers.activation import SiluAndMul
from nanovllm_jax.layers.attention import Attention
from nanovllm_jax.layers.layernorm import RMSNorm
from nanovllm_jax.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm_jax.layers.rotary_embedding import get_rope, RotaryEmbedding
from nanovllm_jax.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm_jax.utils.context import AttentionContext


def divide(numerator: int, denominator: int) -> int:
    """Integer division with assertion that it divides evenly."""
    assert numerator % denominator == 0
    return numerator // denominator


class Qwen3Attention(nnx.Module):
    """Qwen3 attention block with GQA and RoPE.
    
    Supports tensor parallelism via Mesh + shard_map.
    
    Attributes:
        qkv_proj: Fused Q/K/V projection layer.
        o_proj: Output projection layer.
        rotary_emb: Rotary position embeddings.
        attn: Core attention computation.
        q_norm: Optional QK normalization (RMSNorm on Q).
        k_norm: Optional QK normalization (RMSNorm on K).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        block_size: int = 256,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.hidden_size = hidden_size
        self.mesh = mesh
        # Use float32 for model computations - universal GPU compatibility
        self.model_dtype = jnp.float32
        
        # QKV projection (fused)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            mesh=mesh,
            rngs=rngs,
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            mesh=mesh,
            rngs=rngs,
        )
        
        # Rotary embeddings (not an nnx.Module, just computation)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
        )
        
        # Attention
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            block_size=block_size,
        )
        
        # QK normalization (when no qkv_bias)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
    
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Forward pass for attention.
        
        Args:
            positions: Position indices [num_tokens].
            hidden_states: Input tensor [num_tokens, hidden_size].
            context: Attention context with metadata.
        
        Returns:
            Output tensor [num_tokens, hidden_size].
        """
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        
        # Split into Q, K, V
        q, k, v = jnp.split(qkv, [self.q_size, self.q_size + self.kv_size], axis=-1)
        
        # Reshape to [num_tokens, num_heads, head_dim]
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        
        # QK normalization (if no bias)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)
        # Cast to model dtype for consistent compute with KV cache
        q = q.astype(self.model_dtype)
        k = k.astype(self.model_dtype)
        v = v.astype(self.model_dtype)
        
        # Attention
        o = self.attn(q, k, v, context)
        
        # Reshape and project output
        o = o.reshape(-1, self.num_heads * self.head_dim)
        output = self.o_proj(o)
        
        return output


class Qwen3MLP(nnx.Module):
    """Qwen3 MLP block with SwiGLU activation.
    
    Architecture: gate_up_proj -> SiLU(gate) * up -> down_proj
    Supports tensor parallelism via Mesh + shard_map.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        tp_size: int = 1,
        tp_rank: int = 0,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        assert hidden_act == "silu", f"Only silu activation supported, got {hidden_act}"
        
        # Fused gate + up projection
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            mesh=mesh,
            rngs=rngs,
        )
        
        # Down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            mesh=mesh,
            rngs=rngs,
        )
        
        self.act_fn = SiluAndMul()
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.
        
        Args:
            x: Input tensor [num_tokens, hidden_size].
        
        Returns:
            Output tensor [num_tokens, hidden_size].
        """
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nnx.Module):
    """Single Qwen3 transformer decoder layer.
    
    Architecture:
        x = x + attention(layernorm(x))
        x = x + mlp(layernorm(x))
    
    Uses pre-norm architecture with fused residual handling.
    Supports tensor parallelism via Mesh + shard_map.
    """
    
    def __init__(
        self,
        config: Qwen3Config,
        tp_size: int = 1,
        tp_rank: int = 0,
        block_size: int = 256,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            tp_size=tp_size,
            tp_rank=tp_rank,
            block_size=block_size,
            mesh=mesh,
            rngs=rngs,
        )
        
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_size=tp_size,
            tp_rank=tp_rank,
            mesh=mesh,
            rngs=rngs,
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        residual: jax.Array | None,
        context: AttentionContext,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass for decoder layer.
        
        Args:
            positions: Position indices [num_tokens].
            hidden_states: Input tensor [num_tokens, hidden_size].
            residual: Residual tensor from previous layer (None for first layer).
            context: Attention context.
        
        Returns:
            Tuple of (output, residual) for next layer.
        """
        # Pre-attention norm with residual fusion
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Self attention
        hidden_states = self.self_attn(positions, hidden_states, context)
        
        # Post-attention norm with residual fusion
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class Qwen3Model(nnx.Module):
    """Qwen3 transformer model (without LM head).
    
    Stack of embedding + decoder layers + final norm.
    Supports tensor parallelism via Mesh + shard_map.
    """
    
    def __init__(
        self,
        config: Qwen3Config,
        tp_size: int = 1,
        tp_rank: int = 0,
        block_size: int = 256,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.mesh = mesh
        
        # Token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            rngs=rngs,
        )
        
        # Decoder layers
        self.layers = [
            Qwen3DecoderLayer(
                config,
                tp_size=tp_size,
                tp_rank=tp_rank,
                block_size=block_size,
                mesh=mesh,
                rngs=rngs,
            )
            for _ in range(config.num_hidden_layers)
        ]
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Forward pass through transformer.
        
        Args:
            input_ids: Token IDs [num_tokens].
            positions: Position indices [num_tokens].
            context: Attention context.
        
        Returns:
            Hidden states [num_tokens, hidden_size].
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Pass through decoder layers
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, context)
        
        # Final norm
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class Qwen3ForCausalLM(nnx.Module):
    """Qwen3 model with language modeling head.
    
    This is the main model class for text generation.
    Supports tensor parallelism via Mesh + shard_map.
    
    Attributes:
        packed_modules_mapping: Maps HuggingFace weight names to fused layer names.
            Used by the weight loader to handle fused QKV and gate/up projections.
    """
    
    # Maps HuggingFace weight names to our fused layer names
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(
        self,
        config: Qwen3Config,
        tp_size: int = 1,
        tp_rank: int = 0,
        block_size: int = 256,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.mesh = mesh
        
        # Transformer model
        self.model = Qwen3Model(
            config,
            tp_size=tp_size,
            tp_rank=tp_rank,
            block_size=block_size,
            mesh=mesh,
            rngs=rngs,
        )
        
        # LM head
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            rngs=rngs,
        )
        
        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Forward pass (returns hidden states, not logits).
        
        Args:
            input_ids: Token IDs [num_tokens].
            positions: Position indices [num_tokens].
            context: Attention context.
        
        Returns:
            Hidden states [num_tokens, hidden_size].
        """
        return self.model(input_ids, positions, context)
    
    def compute_logits(
        self,
        hidden_states: jax.Array,
        context: AttentionContext,
    ) -> jax.Array:
        """Compute vocabulary logits from hidden states.
        
        In prefill mode, only computes logits for the last token of each sequence.
        
        Args:
            hidden_states: Hidden states [num_tokens, hidden_size].
            context: Attention context (used for last token indices in prefill).
        
        Returns:
            Logits [batch_size, vocab_size].
        """
        return self.lm_head(
            hidden_states,
            last_token_indices=context.last_token_indices if context.is_prefill else None,
        )
