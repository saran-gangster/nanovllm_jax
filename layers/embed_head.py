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
    
    For simplicity in JAX, we replicate the full embedding table on each device
    when tp_size > 1. This avoids the complexity of all-reduce operations
    that require shard_map contexts.
    
    The weight loader will handle loading the appropriate shard if needed,
    but for inference, replication is simpler and embedding lookup is fast.
    
    Attributes:
        num_embeddings: Total vocabulary size.
        embedding_dim: Dimension of embeddings.
        tp_size: Tensor parallel world size (kept for API compatibility).
        tp_rank: This device's rank (kept for API compatibility).
        weight: Embedding weight of shape [vocab_size, embedding_dim].
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
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Keep full embedding table (replicated across devices)
        # This is simpler and avoids all-reduce complexity
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (num_embeddings, embedding_dim)) * 0.01
        )
    
    def load_weights(self, loaded_weight: jax.Array):
        """Load weights (full table, replicated).
        
        Args:
            loaded_weight: Full embedding table of shape [vocab_size, embedding_dim].
        """
        self.weight.value = loaded_weight
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Look up embeddings for input token IDs.
        
        Args:
            x: Token IDs of shape [batch_size, seq_len] or [num_tokens].
        
        Returns:
            Embeddings of shape [..., embedding_dim].
        """
        return self.weight.value[x]


class ParallelLMHead(nnx.Module):
    """Language model head with vocabulary parallelism.
    
    Projects hidden states to vocabulary logits. For simplicity in JAX,
    we replicate the full projection weight on each device when tp_size > 1.
    
    In prefill mode, only processes the last token of each sequence.
    
    Attributes:
        num_embeddings: Total vocabulary size.
        embedding_dim: Input hidden dimension.
        tp_size: Tensor parallel world size (kept for API compatibility).
        tp_rank: This device's rank (kept for API compatibility).
        weight: Projection weight of shape [vocab_size, embedding_dim].
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
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Keep full weight (replicated across devices)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (num_embeddings, embedding_dim)) * 0.01
        )
    
    def load_weights(self, loaded_weight: jax.Array):
        """Load weights (full table, replicated).
        
        Args:
            loaded_weight: Full LM head weight of shape [vocab_size, embedding_dim].
        """
        self.weight.value = loaded_weight
    
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
            Logits of shape [batch_size, vocab_size].
        """
        # In prefill mode, only process last token of each sequence
        if last_token_indices is not None:
            x = x[last_token_indices]
        
        # Full projection: [batch, hidden] @ [hidden, vocab] -> [batch, vocab]
        logits = x @ self.weight.value.T
        
        return logits
