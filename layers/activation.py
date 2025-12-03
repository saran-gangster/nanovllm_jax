"""Activation functions for JAX.

Provides activation functions commonly used in transformer MLPs.
"""

import jax
import jax.numpy as jnp
from jax.nn import silu
from flax import nnx


class SiluAndMul(nnx.Module):
    """SiLU activation with element-wise multiplication.
    
    Used in Qwen3/Llama style MLPs where gate_proj and up_proj outputs
    are concatenated and this activation splits and combines them:
        output = SiLU(gate) * up
    
    This is also known as SwiGLU when the gate uses sigmoid.
    """
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply SiLU-and-mul activation.
        
        Args:
            x: Input tensor of shape [..., 2 * intermediate_size].
               First half is gate, second half is up projection.
        
        Returns:
            Output tensor of shape [..., intermediate_size].
        """
        gate, up = jnp.split(x, 2, axis=-1)
        return silu(gate) * up
