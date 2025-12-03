"""Rotary Position Embedding (RoPE) for JAX.

Implements rotary position embeddings as described in the RoFormer paper.
The implementation precomputes cos/sin caches for efficiency.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from functools import lru_cache, partial


@jax.jit
def apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply rotary embeddings to input tensor (JIT compiled).
    
    Args:
        x: Input tensor of shape [..., head_dim].
        cos: Cosine values of shape [..., head_dim // 2].
        sin: Sine values of shape [..., head_dim // 2].
    
    Returns:
        Tensor with rotary embeddings applied.
    """
    # Split into two halves
    # Keep in original dtype (bfloat16) - XLA handles precision internally
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    # Apply rotation (cos/sin are float32, but XLA will optimize the mixed precision)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    return jnp.concatenate([y1, y2], axis=-1)


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding module.
    
    Precomputes and caches cos/sin values for all positions up to max_position_embeddings.
    
    Attributes:
        head_size: Dimension of each attention head.
        cos_sin_cache: Precomputed [cos, sin] values of shape [max_pos, 1, head_dim].
    """
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        assert rotary_dim == head_size, "rotary_dim must equal head_size"
        self.head_size = head_size
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
        
        # Compute position indices
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        
        # Compute frequencies: [max_pos, rotary_dim // 2]
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        
        # Compute cos and sin
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        # Concatenate and add head dimension: [max_pos, 1, head_dim]
        cache = jnp.concatenate([cos, sin], axis=-1)[:, None, :]
        
        # Store as non-trainable variable
        self.cos_sin_cache = nnx.Variable(cache)
    
    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply rotary embeddings to query and key.
        
        Args:
            positions: Position indices of shape [batch] or [num_tokens].
            query: Query tensor of shape [num_tokens, num_heads, head_dim] or 
                   [batch, seq_len, num_heads, head_dim].
            key: Key tensor with same shape as query.
        
        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as inputs.
        """
        # Index into cache: [num_tokens, 1, head_dim] or [batch, 1, head_dim]
        cos_sin = self.cos_sin_cache.value[positions]
        
        # Split into cos and sin
        cos, sin = jnp.split(cos_sin, 2, axis=-1)
        
        # Apply rotary embeddings
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        return query, key


@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    """Get or create a cached RotaryEmbedding instance.
    
    Args:
        head_size: Dimension of each attention head.
        rotary_dim: Dimension of rotary embeddings (must equal head_size).
        max_position: Maximum sequence length.
        base: Base for computing inverse frequencies.
        rope_scaling: Scaling configuration (not supported yet).
    
    Returns:
        RotaryEmbedding instance.
    """
    assert rope_scaling is None, "rope_scaling is not yet supported"
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
