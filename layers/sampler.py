"""Sampler for token generation.

Implements temperature-based sampling with the Gumbel-softmax trick
for efficient parallel sampling.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


class Sampler(nnx.Module):
    """Temperature-based token sampler.
    
    Uses the Gumbel-max trick for sampling from categorical distributions,
    which is equivalent to sampling from softmax but more numerically stable.
    """
    
    def __init__(self, *, rngs: nnx.Rngs):
        self.rngs = rngs
    
    @partial(nnx.jit, static_argnums=())
    def __call__(
        self,
        logits: jax.Array,
        temperatures: jax.Array,
    ) -> jax.Array:
        """Sample tokens from logits with per-sequence temperatures.
        
        Args:
            logits: Logits tensor of shape [batch_size, vocab_size].
            temperatures: Temperature values of shape [batch_size].
        
        Returns:
            Sampled token IDs of shape [batch_size].
        """
        # Scale logits by temperature (use float32 for numerical stability)
        logits = logits.astype(jnp.float32) / temperatures[:, None].astype(jnp.float32)
        
        # Gumbel-max trick for sampling
        # sample = argmax(logits + gumbel_noise)
        # where gumbel_noise = -log(-log(uniform))
        key = self.rngs()
        uniform = jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)
        gumbel_noise = -jnp.log(-jnp.log(uniform))
        
        sample_tokens = jnp.argmax(logits + gumbel_noise, axis=-1)
        
        return sample_tokens
