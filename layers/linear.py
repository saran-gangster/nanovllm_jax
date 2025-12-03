"""Tensor-parallel linear layers for JAX with Flax NNX.

This module provides linear layers that support tensor parallelism across multiple
GPUs using JAX's sharding primitives with shard_map for explicit SPMD control.

Key classes:
- ReplicatedLinear: Non-sharded linear layer (weights replicated across devices)
- ColumnParallelLinear: Shards output dimension across devices
- RowParallelLinear: Shards input dimension across devices, with all-reduce
- QKVParallelLinear: Fused Q/K/V projection with proper head-wise sharding
- MergedColumnParallelLinear: Fused gate/up projections for MLP

Tensor Parallelism Approach:
- Uses jax.sharding.Mesh + NamedSharding for device placement
- Uses jax.experimental.shard_map for explicit SPMD programming
- All-reduce via jax.lax.psum with axis_name="tp"
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from flax import nnx
from typing import Literal
from functools import partial


def divide(numerator: int, denominator: int) -> int:
    """Integer division with assertion that it divides evenly."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class LinearBase(nnx.Module):
    """Base class for all linear layers with tensor parallelism support.
    
    Uses jax.sharding.Mesh + NamedSharding with shard_map for explicit TP.
    
    Attributes:
        input_size: Input feature dimension.
        output_size: Output feature dimension (after TP sharding if applicable).
        tp_size: Tensor parallel world size.
        tp_rank: This device's rank in tensor parallel group.
        tp_dim: Which dimension to shard (0=output, 1=input, None=replicated).
        mesh: JAX device mesh for sharding (optional, for TP > 1).
        weight: Weight parameter of shape [output_size, input_size].
        bias: Optional bias parameter of shape [output_size].
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_dim: int | None = None,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_dim = tp_dim
        self.mesh = mesh
        
        # Initialize weight with small values (will be overwritten by loader)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (output_size, input_size)) * 0.01
        )
        
        if bias:
            self.bias = nnx.Param(jnp.zeros((output_size,)))
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Linear layer with weights replicated across all devices.
    
    No tensor parallelism - same weights on all devices.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(input_size, output_size, bias, tp_size, tp_rank, None, mesh, rngs=rngs)
    
    def load_weights(self, loaded_weight: jax.Array, loaded_bias: jax.Array | None = None):
        """Load weights without sharding."""
        self.weight.value = loaded_weight
        if loaded_bias is not None and self.bias is not None:
            self.bias.value = loaded_bias
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: y = xW^T + b"""
        y = x @ self.weight.value.T
        if self.bias is not None:
            y = y + self.bias.value
        return y


class ColumnParallelLinear(LinearBase):
    """Linear layer with column (output) parallelism using shard_map.
    
    The weight matrix is sharded along the output dimension (dim 0).
    Each device holds a slice of shape [output_size // tp_size, input_size].
    No all-reduce needed since outputs are independent.
    
    With shard_map, the matmul runs independently on each device with its
    local weight shard, producing sharded outputs.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        # Each device holds output_size // tp_size rows
        sharded_output_size = divide(output_size, tp_size)
        super().__init__(
            input_size, sharded_output_size, bias, tp_size, tp_rank, 0, mesh, rngs=rngs
        )
        self.full_output_size = output_size
    
    def load_weights(self, loaded_weight: jax.Array, loaded_bias: jax.Array | None = None):
        """Load weights with column sharding.
        
        Args:
            loaded_weight: Full weight tensor of shape [output_size, input_size].
            loaded_bias: Optional full bias tensor of shape [output_size].
        """
        shard_size = self.output_size  # Already divided
        start_idx = self.tp_rank * shard_size
        
        # Slice output dimension
        self.weight.value = lax.dynamic_slice(
            loaded_weight,
            (start_idx, 0),
            (shard_size, self.input_size)
        )
        
        if loaded_bias is not None and self.bias is not None:
            self.bias.value = lax.dynamic_slice(
                loaded_bias,
                (start_idx,),
                (shard_size,)
            )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: y = xW^T + b (local slice).
        
        When using shard_map, this runs on each device with its local weight shard.
        The output is sharded along the last dimension.
        """
        y = x @ self.weight.value.T
        if self.bias is not None:
            y = y + self.bias.value
        return y
    
    def sharded_call(self, x: jax.Array) -> jax.Array:
        """Forward pass using explicit shard_map for TP > 1.
        
        This method explicitly uses shard_map for column-parallel matmul.
        Input is replicated, output is sharded along tp axis.
        """
        if self.mesh is None or self.tp_size <= 1:
            return self(x)
        
        @partial(shard_map, mesh=self.mesh,
                 in_specs=(P(), P("tp", None)),  # x replicated, weight sharded on dim 0
                 out_specs=P(None, "tp"))  # output sharded on last dim
        def column_matmul(x_local: jax.Array, weight_local: jax.Array) -> jax.Array:
            return x_local @ weight_local.T
        
        y = column_matmul(x, self.weight.value)
        if self.bias is not None:
            # Bias is also sharded along output dim
            y = y + self.bias.value
        return y


class MergedColumnParallelLinear(LinearBase):
    """Fused column-parallel linear for multiple output projections.
    
    Used for fusing gate_proj and up_proj in MLP (Qwen3/Llama style).
    Each projection is independently sharded along the output dimension.
    
    Example: gate_proj (hidden -> ffn_dim) and up_proj (hidden -> ffn_dim)
    are fused into a single [2 * ffn_dim, hidden] weight matrix.
    
    With shard_map, each device holds a portion of both projections.
    """
    
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.output_sizes = output_sizes  # [ffn_dim, ffn_dim] for gate+up
        total_output = sum(output_sizes)
        sharded_output_size = divide(total_output, tp_size)
        
        super().__init__(
            input_size, sharded_output_size, bias, tp_size, tp_rank, 0, mesh, rngs=rngs
        )
        self.full_output_sizes = output_sizes
    
    def load_weights(
        self,
        loaded_weight: jax.Array,
        shard_id: int,
        loaded_bias: jax.Array | None = None,
    ):
        """Load weights for one of the merged projections.
        
        Args:
            loaded_weight: Weight tensor for this projection [output_size_i, input_size].
            shard_id: Which projection (0=gate, 1=up for MLP).
            loaded_bias: Optional bias for this projection.
        """
        # Calculate offset within merged weight
        shard_offset = sum(self.full_output_sizes[:shard_id]) // self.tp_size
        shard_size = self.full_output_sizes[shard_id] // self.tp_size
        
        # Get this device's slice of the loaded weight
        tp_shard_size = loaded_weight.shape[0] // self.tp_size
        start_idx = self.tp_rank * tp_shard_size
        weight_slice = lax.dynamic_slice(
            loaded_weight,
            (start_idx, 0),
            (tp_shard_size, self.input_size)
        )
        
        # Update the appropriate slice of our weight
        self.weight.value = self.weight.value.at[shard_offset:shard_offset + shard_size, :].set(
            weight_slice
        )
        
        if loaded_bias is not None and self.bias is not None:
            bias_slice = lax.dynamic_slice(loaded_bias, (start_idx,), (tp_shard_size,))
            self.bias.value = self.bias.value.at[shard_offset:shard_offset + shard_size].set(
                bias_slice
            )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: y = xW^T + b
        
        Each device computes its local shard of the merged projections.
        """
        y = x @ self.weight.value.T
        if self.bias is not None:
            y = y + self.bias.value
        return y
    
    def sharded_call(self, x: jax.Array) -> jax.Array:
        """Forward pass using explicit shard_map for TP > 1."""
        if self.mesh is None or self.tp_size <= 1:
            return self(x)
        
        @partial(shard_map, mesh=self.mesh,
                 in_specs=(P(), P("tp", None)),
                 out_specs=P(None, "tp"))
        def merged_column_matmul(x_local: jax.Array, weight_local: jax.Array) -> jax.Array:
            return x_local @ weight_local.T
        
        y = merged_column_matmul(x, self.weight.value)
        if self.bias is not None:
            y = y + self.bias.value
        return y


class QKVParallelLinear(LinearBase):
    """Fused Q/K/V projection with head-wise tensor parallelism using shard_map.
    
    Fuses query, key, and value projections into a single weight matrix.
    Sharding is done per-head, so each device gets a subset of attention heads.
    
    Weight layout: [q_size + k_size + v_size, hidden_size]
    where q_size = num_heads * head_dim, k_size = v_size = num_kv_heads * head_dim
    
    With shard_map, each device computes QKV for its local heads.
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        
        # Output size per device: q_heads + k_heads + v_heads (all sharded)
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        
        super().__init__(
            hidden_size, output_size, bias, tp_size, tp_rank, 0, mesh, rngs=rngs
        )
    
    def load_weights(
        self,
        loaded_weight: jax.Array,
        shard_id: Literal["q", "k", "v"],
        loaded_bias: jax.Array | None = None,
    ):
        """Load weights for Q, K, or V projection.
        
        Args:
            loaded_weight: Full weight for this projection [proj_size, hidden_size].
            shard_id: Which projection ("q", "k", or "v").
            loaded_bias: Optional bias for this projection.
        """
        # Determine offset and size in the fused weight
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # Get this device's slice from loaded weight
        full_shard_size = loaded_weight.shape[0] // self.tp_size
        start_idx = self.tp_rank * full_shard_size
        weight_slice = lax.dynamic_slice(
            loaded_weight,
            (start_idx, 0),
            (full_shard_size, self.input_size)
        )
        
        # Update the appropriate slice
        self.weight.value = self.weight.value.at[shard_offset:shard_offset + shard_size, :].set(
            weight_slice
        )
        
        if loaded_bias is not None and self.bias is not None:
            bias_slice = lax.dynamic_slice(loaded_bias, (start_idx,), (full_shard_size,))
            self.bias.value = self.bias.value.at[shard_offset:shard_offset + shard_size].set(
                bias_slice
            )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: returns concatenated [Q, K, V] for this device's heads.
        
        Each device computes QKV for its local subset of attention heads.
        """
        y = x @ self.weight.value.T
        if self.bias is not None:
            y = y + self.bias.value
        return y
    
    def sharded_call(self, x: jax.Array) -> jax.Array:
        """Forward pass using explicit shard_map for TP > 1."""
        if self.mesh is None or self.tp_size <= 1:
            return self(x)
        
        @partial(shard_map, mesh=self.mesh,
                 in_specs=(P(), P("tp", None)),
                 out_specs=P(None, "tp"))
        def qkv_matmul(x_local: jax.Array, weight_local: jax.Array) -> jax.Array:
            return x_local @ weight_local.T
        
        y = qkv_matmul(x, self.weight.value)
        if self.bias is not None:
            y = y + self.bias.value
        return y


class RowParallelLinear(LinearBase):
    """Linear layer with row (input) parallelism using shard_map.
    
    The weight matrix is sharded along the input dimension (dim 1).
    Each device holds a slice of shape [output_size, input_size // tp_size].
    Requires all-reduce (psum) after matmul to sum partial results.
    
    With shard_map, each device computes partial output and then all-reduce
    combines the results across the "tp" axis.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        # Each device holds input_size // tp_size columns
        sharded_input_size = divide(input_size, tp_size)
        super().__init__(
            sharded_input_size, output_size, bias, tp_size, tp_rank, 1, mesh, rngs=rngs
        )
        self.full_input_size = input_size
    
    def load_weights(self, loaded_weight: jax.Array, loaded_bias: jax.Array | None = None):
        """Load weights with row sharding.
        
        Args:
            loaded_weight: Full weight tensor of shape [output_size, input_size].
            loaded_bias: Optional full bias tensor of shape [output_size].
        """
        shard_size = self.input_size  # Already divided
        start_idx = self.tp_rank * shard_size
        
        # Slice input dimension (columns)
        self.weight.value = lax.dynamic_slice(
            loaded_weight,
            (0, start_idx),
            (self.output_size, shard_size)
        )
        
        # Bias is NOT sharded - only rank 0 adds it (or we can add full bias on all)
        if loaded_bias is not None and self.bias is not None:
            self.bias.value = loaded_bias
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with all-reduce.
        
        Args:
            x: Input with last dim = input_size // tp_size (already sharded).
        
        Returns:
            Output after all-reduce across TP group.
        
        Note: When tp_size > 1, this assumes we're inside a shard_map context
        where psum is valid. Use sharded_call for explicit shard_map.
        """
        y = x @ self.weight.value.T
        
        # All-reduce across tensor parallel group
        # Note: psum only works inside shard_map or pmap with axis_name
        if self.tp_size > 1:
            y = lax.psum(y, axis_name="tp")
        
        # Add bias after all-reduce (only once)
        if self.bias is not None:
            y = y + self.bias.value
        
        return y
    
    def sharded_call(self, x: jax.Array) -> jax.Array:
        """Forward pass using explicit shard_map with all-reduce.
        
        Input is sharded (from column-parallel), output is replicated after psum.
        """
        if self.mesh is None or self.tp_size <= 1:
            y = x @ self.weight.value.T
            if self.bias is not None:
                y = y + self.bias.value
            return y
        
        @partial(shard_map, mesh=self.mesh,
                 in_specs=(P(None, "tp"), P(None, "tp")),  # x sharded on last dim, weight sharded on dim 1
                 out_specs=P(),  # output replicated after all-reduce
                 check_rep=False)  # Output needs reduction
        def row_matmul_with_reduce(x_local: jax.Array, weight_local: jax.Array) -> jax.Array:
            y_partial = x_local @ weight_local.T
            return lax.psum(y_partial, axis_name="tp")
        
        y = row_matmul_with_reduce(x, self.weight.value)
        
        if self.bias is not None:
            y = y + self.bias.value
        
        return y
