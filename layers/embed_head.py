"""Embedding and LM head layers for JAX with tensor parallelism.

Provides vocabulary-parallel embeddings and language model head that
shard the vocabulary dimension across tensor parallel devices.
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
from dataclasses import dataclass


def divide(numerator: int, denominator: int) -> int:
    """Integer division with assertion that it divides evenly."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class VocabParallelEmbedding(nnx.Module):
    """Embedding layer with vocabulary parallelism.
    
    The embedding table is sharded along the vocabulary dimension.
    Each device holds vocab_size // tp_size rows.
    
    For tokens outside this device's range, we return zeros and use
    all-reduce to combine results from all devices.
    
    Attributes:
        num_embeddings: Total vocabulary size.
        embedding_dim: Dimension of embeddings.
        tp_size: Tensor parallel world size.
        tp_rank: This device's rank.
        vocab_start_idx: Start of this device's vocabulary range.
        vocab_end_idx: End of this device's vocabulary range.
        weight: Embedding weight of shape [vocab_size // tp_size, embedding_dim].
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        assert num_embeddings % tp_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by tp_size ({tp_size})"
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # Initialize with small random values (will be overwritten by loader)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (self.num_embeddings_per_partition, embedding_dim)) * 0.01
        )
    
    def load_weights(self, loaded_weight: jax.Array):
        """Load weights with vocabulary sharding.
        
        Args:
            loaded_weight: Full embedding table of shape [vocab_size, embedding_dim].
        """
        shard_size = self.num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        
        self.weight.value = lax.dynamic_slice(
            loaded_weight,
            (start_idx, 0),
            (shard_size, self.embedding_dim)
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Look up embeddings for input token IDs.
        
        Args:
            x: Token IDs of shape [batch_size, seq_len] or [num_tokens].
        
        Returns:
            Embeddings of shape [..., embedding_dim].
        """
        if self.tp_size > 1:
            # Create mask for tokens in this device's range
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Adjust indices to local range
            local_x = jnp.where(mask, x - self.vocab_start_idx, 0)
        else:
            local_x = x
            mask = None
        
        # Look up embeddings
        y = self.weight.value[local_x]
        
        if self.tp_size > 1:
            # Zero out embeddings for tokens not in this device's range
            y = jnp.where(mask[..., None], y, 0.0)
            # All-reduce to combine from all devices
            y = lax.psum(y, axis_name="tp")
        
        return y


class ParallelLMHead(nnx.Module):
    """Language model head with vocabulary parallelism.
    
    Projects hidden states to vocabulary logits. The projection is sharded
    across devices along the vocabulary dimension.
    
    In prefill mode, only processes the last token of each sequence.
    After local projection, gathers results from all devices to rank 0.
    
    Attributes:
        num_embeddings: Total vocabulary size.
        embedding_dim: Input hidden dimension.
        tp_size: Tensor parallel world size.
        tp_rank: This device's rank.
        weight: Projection weight of shape [vocab_size // tp_size, embedding_dim].
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        assert num_embeddings % tp_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by tp_size ({tp_size})"
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        self.num_embeddings_per_partition = num_embeddings // tp_size
        
        # Initialize with small random values (will be overwritten by loader)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (self.num_embeddings_per_partition, embedding_dim)) * 0.01
        )
    
    def load_weights(self, loaded_weight: jax.Array):
        """Load weights with vocabulary sharding.
        
        Args:
            loaded_weight: Full LM head weight of shape [vocab_size, embedding_dim].
        """
        shard_size = self.num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        
        self.weight.value = lax.dynamic_slice(
            loaded_weight,
            (start_idx, 0),
            (shard_size, self.embedding_dim)
        )
    
    def __call__(
        self,
        x: jax.Array,
        last_token_indices: jax.Array | None = None,
    ) -> jax.Array:
        """Project hidden states to vocabulary logits.
        
        Args:
            x: Hidden states of shape [num_tokens, hidden_dim] or [batch, seq_len, hidden_dim].
            last_token_indices: For prefill, indices of last tokens to process.
                If None, processes all tokens.
        
        Returns:
            Logits of shape [batch_size, vocab_size] on rank 0, None on other ranks.
        """
        # In prefill mode, only process last token of each sequence
        if last_token_indices is not None:
            x = x[last_token_indices]
        
        # Local projection: [batch, hidden] @ [hidden, vocab_shard] -> [batch, vocab_shard]
        logits = x @ self.weight.value.T
        
        if self.tp_size > 1:
            # All-gather logits from all devices to get full vocabulary
            # This gathers along the last dimension
            all_logits = lax.all_gather(logits, axis_name="tp", axis=-1, tiled=True)
            return all_logits
        
        return logits
