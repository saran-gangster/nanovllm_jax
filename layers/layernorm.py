"""Layer normalization implementations for JAX.

Provides RMSNorm (Root Mean Square Layer Normalization) used in modern
transformer architectures like Llama and Qwen.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.
    
    Normalizes inputs by their RMS value, without centering (no mean subtraction).
    More efficient than LayerNorm for transformers.
    
    Attributes:
        eps: Small constant for numerical stability.
        weight: Learnable scale parameter of shape [hidden_size].
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((hidden_size,)))
    
    def _rms_forward(self, x: jax.Array) -> jax.Array:
        """Apply RMS normalization without residual addition.
        
        Args:
            x: Input tensor of shape [..., hidden_size].
        
        Returns:
            Normalized tensor of same shape.
        """
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        
        # Compute RMS
        var = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        
        # Scale and cast back
        return (x * self.weight.value).astype(orig_dtype)
    
    def _add_rms_forward(
        self,
        x: jax.Array,
        residual: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply RMS normalization with residual addition.
        
        Fuses residual addition with normalization for efficiency.
        
        Args:
            x: Input tensor of shape [..., hidden_size].
            residual: Residual tensor of same shape.
        
        Returns:
            Tuple of (normalized_output, updated_residual).
        """
        orig_dtype = x.dtype
        
        # Add residual
        x = x.astype(jnp.float32) + residual.astype(jnp.float32)
        residual = x.astype(orig_dtype)
        
        # Compute RMS
        var = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        
        # Scale and cast back
        x = (x * self.weight.value).astype(orig_dtype)
        
        return x, residual
    
    def __call__(
        self,
        x: jax.Array,
        residual: jax.Array | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [..., hidden_size].
            residual: Optional residual tensor for fused add-norm.
        
        Returns:
            If residual is None: normalized tensor.
            If residual is provided: tuple of (normalized, updated_residual).
        """
        if residual is None:
            return self._rms_forward(x)
        else:
            return self._add_rms_forward(x, residual)
    
    def load_weights(self, loaded_weight: jax.Array):
        """Load weight from checkpoint."""
        self.weight.value = loaded_weight
