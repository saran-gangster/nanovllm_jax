"""Attention context for passing metadata through the model.

In the PyTorch version, this uses global state. In JAX, we pass context
explicitly through function calls for purity.

The AttentionContext is registered as a JAX PyTree so it can be passed
through JIT-compiled functions.

OPTIMIZATION: Bucketed max_seqlen values (powers of 2) reduce JIT
recompilations from O(unique_lengths) to O(log(max_length)).
"""

from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp


def _bucket_size(n: int, min_size: int = 16) -> int:
    """Round up to next power of 2 for JIT cache efficiency."""
    if n <= min_size:
        return min_size
    return 2 ** math.ceil(math.log2(n))


@dataclass
class AttentionContext:
    """Context containing all metadata needed for attention computation.
    
    This is passed explicitly to the model's forward function rather than
    using global state, maintaining JAX's functional paradigm.
    
    Registered as a JAX PyTree for JIT compatibility.
    
    Attributes:
        is_prefill: True for prefill phase (processing prompt), False for decode.
        
        # Prefill-specific (variable-length sequences):
        cu_seqlens_q: Cumulative sequence lengths for queries [num_seqs + 1].
        cu_seqlens_k: Cumulative sequence lengths for keys [num_seqs + 1].
        max_seqlen_q: Maximum query sequence length (BUCKETED to power of 2).
        max_seqlen_k: Maximum key sequence length (BUCKETED to power of 2).
        
        # Decode-specific (single token per sequence):
        context_lens: Current context length for each sequence [batch_size].
        
        # Shared:
        slot_mapping: Maps token positions to KV cache slots [num_tokens].
                     -1 indicates cached token (don't store).
        block_tables: Block table for paged attention [batch_size, max_blocks].
                     Maps logical blocks to physical block IDs.
        
        # For LM head in prefill:
        last_token_indices: Indices of last tokens in packed sequence [batch_size].
    """
    is_prefill: bool = False
    
    # Prefill metadata - max_seqlen values are BUCKETED to powers of 2
    cu_seqlens_q: jnp.ndarray | None = None
    cu_seqlens_k: jnp.ndarray | None = None
    max_seqlen_q: int = 0  # Bucketed to power of 2
    max_seqlen_k: int = 0  # Bucketed to power of 2
    
    # Decode metadata  
    context_lens: jnp.ndarray | None = None
    
    # Shared metadata
    slot_mapping: jnp.ndarray | None = None
    block_tables: jnp.ndarray | None = None
    
    # LM head metadata
    last_token_indices: jnp.ndarray | None = None


# Register AttentionContext as a JAX PyTree
# Static fields (not traced): is_prefill, max_seqlen_q, max_seqlen_k
# Dynamic fields (traced as arrays): cu_seqlens_q, cu_seqlens_k, context_lens,
#                                     slot_mapping, block_tables, last_token_indices
def _attention_context_flatten(ctx):
    """Flatten AttentionContext for PyTree registration."""
    # Children are the dynamic (array) fields
    children = (
        ctx.cu_seqlens_q,
        ctx.cu_seqlens_k,
        ctx.context_lens,
        ctx.slot_mapping,
        ctx.block_tables,
        ctx.last_token_indices,
    )
    # Aux data is the static fields (bucketed max_seqlen reduces recompilations)
    aux_data = (ctx.is_prefill, ctx.max_seqlen_q, ctx.max_seqlen_k)
    return children, aux_data


def _attention_context_unflatten(aux_data, children):
    """Unflatten to reconstruct AttentionContext."""
    is_prefill, max_seqlen_q, max_seqlen_k = aux_data
    cu_seqlens_q, cu_seqlens_k, context_lens, slot_mapping, block_tables, last_token_indices = children
    return AttentionContext(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        context_lens=context_lens,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        last_token_indices=last_token_indices,
    )


# Register the PyTree
jax.tree_util.register_pytree_node(
    AttentionContext,
    _attention_context_flatten,
    _attention_context_unflatten,
)


def create_prefill_context(
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    slot_mapping: jnp.ndarray,
    block_tables: jnp.ndarray | None = None,
) -> AttentionContext:
    """Create context for prefill phase.
    
    Args:
        cu_seqlens_q: Cumulative query sequence lengths [num_seqs + 1].
        cu_seqlens_k: Cumulative key sequence lengths [num_seqs + 1].
        max_seqlen_q: Maximum query length in batch.
        max_seqlen_k: Maximum key length in batch.
        slot_mapping: Token to KV cache slot mapping.
        block_tables: Optional block tables for prefix caching.
    
    Returns:
        AttentionContext configured for prefill.
    """
    # Compute last token indices for LM head
    last_token_indices = cu_seqlens_q[1:] - 1
    
    # OPTIMIZATION: Bucket sizes to powers of 2 to reduce JIT recompilations
    bucketed_q = _bucket_size(max_seqlen_q)
    bucketed_k = _bucket_size(max_seqlen_k)
    
    return AttentionContext(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=bucketed_q,  # Bucketed (static)
        max_seqlen_k=bucketed_k,  # Bucketed (static)
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        last_token_indices=last_token_indices,
    )


def create_decode_context(
    context_lens: jnp.ndarray,
    slot_mapping: jnp.ndarray,
    block_tables: jnp.ndarray,
) -> AttentionContext:
    """Create context for decode phase.
    
    Args:
        context_lens: Current context length for each sequence [batch_size].
        slot_mapping: Token to KV cache slot mapping [batch_size].
        block_tables: Block tables for paged attention [batch_size, max_blocks].
    
    Returns:
        AttentionContext configured for decode.
    """
    return AttentionContext(
        is_prefill=False,
        context_lens=context_lens,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        last_token_indices=None,  # Not needed for decode (all tokens are "last")
    )
