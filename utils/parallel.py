"""Tensor Parallelism utilities using JAX Mesh and shard_map.

This module provides explicit tensor parallelism primitives:
- Mesh: Defines device layout for parallelism
- NamedSharding: Specifies how arrays are partitioned across mesh
- shard_map: Explicit SPMD programming for TP operations

Key patterns:
- Column parallelism: Shard output dimension (axis 0 of weight)
- Row parallelism: Shard input dimension (axis 1 of weight), all-reduce output
- All-reduce via jax.lax.psum with axis_name="tp"
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial
from typing import Callable, Any


# =============================================================================
# Mesh Context Management
# =============================================================================

def create_tp_mesh(tp_size: int = 1) -> Mesh:
    """Create a device mesh for tensor parallelism.
    
    Args:
        tp_size: Number of devices for tensor parallelism.
    
    Returns:
        A JAX Mesh with axis "tp" of size tp_size.
    """
    devices = jax.devices()[:tp_size]
    if len(devices) < tp_size:
        raise ValueError(f"Requested {tp_size} devices but only {len(devices)} available")
    return Mesh(devices, axis_names=("tp",))


def get_tp_sharding(mesh: Mesh, spec: P | None = None) -> NamedSharding:
    """Create a NamedSharding for tensor parallelism.
    
    Args:
        mesh: The device mesh.
        spec: PartitionSpec defining sharding. None means replicated.
    
    Returns:
        NamedSharding object for placing arrays.
    """
    return NamedSharding(mesh, spec or P())


# =============================================================================
# Sharding Specifications for Common Patterns
# =============================================================================

# Replicated (same data on all devices)
REPLICATED = P()

# Sharded along first dimension (e.g., output dim of column-parallel)
SHARD_DIM_0 = P("tp")

# Sharded along second dimension (e.g., input dim of row-parallel)
SHARD_DIM_1 = P(None, "tp")

# Sharded along third dimension (e.g., heads in attention)
SHARD_DIM_2 = P(None, None, "tp")


def shard_weight_column(weight: jax.Array, mesh: Mesh) -> jax.Array:
    """Shard a weight matrix along output (column) dimension.
    
    Args:
        weight: Weight of shape [output_size, input_size].
        mesh: Device mesh for sharding.
    
    Returns:
        Sharded weight where each device holds [output_size/tp, input_size].
    """
    sharding = NamedSharding(mesh, P("tp", None))
    return jax.device_put(weight, sharding)


def shard_weight_row(weight: jax.Array, mesh: Mesh) -> jax.Array:
    """Shard a weight matrix along input (row) dimension.
    
    Args:
        weight: Weight of shape [output_size, input_size].
        mesh: Device mesh for sharding.
    
    Returns:
        Sharded weight where each device holds [output_size, input_size/tp].
    """
    sharding = NamedSharding(mesh, P(None, "tp"))
    return jax.device_put(weight, sharding)


def replicate(array: jax.Array, mesh: Mesh) -> jax.Array:
    """Replicate an array across all devices in mesh.
    
    Args:
        array: Array to replicate.
        mesh: Device mesh.
    
    Returns:
        Array replicated on all devices.
    """
    sharding = NamedSharding(mesh, P())
    return jax.device_put(array, sharding)


# =============================================================================
# shard_map Based Tensor Parallel Operations
# =============================================================================

def column_parallel_matmul(
    mesh: Mesh,
    in_specs: P = P(),
    out_specs: P = P("tp"),
) -> Callable:
    """Create a column-parallel matmul using shard_map.
    
    Column parallelism: weight is sharded along output dimension.
    Input is replicated, output is sharded.
    
    Weight: [output_size, input_size] -> each device has [output_size/tp, input_size]
    Input: [batch, input_size] -> replicated
    Output: [batch, output_size/tp] -> sharded, each device has partial output
    
    Args:
        mesh: Device mesh.
        in_specs: Input PartitionSpec (typically replicated).
        out_specs: Output PartitionSpec (typically sharded on tp).
    
    Returns:
        A function that performs column-parallel matmul.
    """
    @partial(shard_map, mesh=mesh, 
             in_specs=(in_specs, P("tp", None)),  # x replicated, weight sharded on dim 0
             out_specs=out_specs)  # output sharded on tp
    def matmul_fn(x: jax.Array, weight: jax.Array) -> jax.Array:
        """x @ weight.T where weight is column-sharded."""
        return x @ weight.T
    
    return matmul_fn


def row_parallel_matmul(
    mesh: Mesh,
    in_specs: P = P("tp"),
    out_specs: P = P(),
    reduce_output: bool = True,
) -> Callable:
    """Create a row-parallel matmul using shard_map.
    
    Row parallelism: weight is sharded along input dimension.
    Input is sharded (from column-parallel output), output needs all-reduce.
    
    Weight: [output_size, input_size] -> each device has [output_size, input_size/tp]
    Input: [batch, input_size/tp] -> sharded
    Output: [batch, output_size] -> needs all-reduce to combine partial sums
    
    Args:
        mesh: Device mesh.
        in_specs: Input PartitionSpec (typically sharded from previous column-parallel).
        out_specs: Output PartitionSpec (typically replicated after all-reduce).
        reduce_output: Whether to all-reduce the output.
    
    Returns:
        A function that performs row-parallel matmul with optional all-reduce.
    """
    @partial(shard_map, mesh=mesh,
             in_specs=(in_specs, P(None, "tp")),  # x sharded, weight sharded on dim 1
             out_specs=out_specs,
             check_rep=False)  # Output needs reduction
    def matmul_fn(x: jax.Array, weight: jax.Array) -> jax.Array:
        """x @ weight.T where weight is row-sharded, with all-reduce."""
        y = x @ weight.T
        if reduce_output:
            y = lax.psum(y, axis_name="tp")
        return y
    
    return matmul_fn


def all_reduce_sum(mesh: Mesh) -> Callable:
    """Create an all-reduce sum operation using shard_map.
    
    Args:
        mesh: Device mesh.
    
    Returns:
        A function that performs all-reduce sum across tp axis.
    """
    @partial(shard_map, mesh=mesh,
             in_specs=P("tp"),
             out_specs=P(),
             check_rep=False)
    def reduce_fn(x: jax.Array) -> jax.Array:
        return lax.psum(x, axis_name="tp")
    
    return reduce_fn


def all_gather(mesh: Mesh, axis: int = 0) -> Callable:
    """Create an all-gather operation using shard_map.
    
    Gathers sharded tensors across devices along specified axis.
    
    Args:
        mesh: Device mesh.
        axis: Axis along which to gather.
    
    Returns:
        A function that performs all-gather.
    """
    in_spec = [None] * (axis + 1)
    in_spec[axis] = "tp"
    
    @partial(shard_map, mesh=mesh,
             in_specs=P(*in_spec),
             out_specs=P(),
             check_rep=False)
    def gather_fn(x: jax.Array) -> jax.Array:
        return lax.all_gather(x, axis_name="tp", axis=axis, tiled=True)
    
    return gather_fn


# =============================================================================
# Tensor Parallel Layer Helpers
# =============================================================================

class TPContext:
    """Context manager for tensor parallelism operations.
    
    Provides mesh context and helper functions for TP operations.
    
    Usage:
        with TPContext(tp_size=4) as tp:
            sharded_weight = tp.shard_column(weight)
            output = tp.column_matmul(x, sharded_weight)
    """
    
    def __init__(self, tp_size: int = 1):
        self.tp_size = tp_size
        self.mesh = create_tp_mesh(tp_size) if tp_size > 1 else None
        self.tp_rank = 0  # Will be set per-device in shard_map
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    @property
    def is_parallel(self) -> bool:
        return self.tp_size > 1
    
    def shard_column(self, weight: jax.Array) -> jax.Array:
        """Shard weight along output (column) dimension."""
        if not self.is_parallel:
            return weight
        return shard_weight_column(weight, self.mesh)
    
    def shard_row(self, weight: jax.Array) -> jax.Array:
        """Shard weight along input (row) dimension."""
        if not self.is_parallel:
            return weight
        return shard_weight_row(weight, self.mesh)
    
    def replicate(self, array: jax.Array) -> jax.Array:
        """Replicate array across all devices."""
        if not self.is_parallel:
            return array
        return replicate(array, self.mesh)


def tp_all_reduce(x: jax.Array, tp_size: int) -> jax.Array:
    """All-reduce across tensor parallel group.
    
    This is a simple wrapper that uses psum when inside shard_map context.
    For use within sharded functions.
    
    Args:
        x: Input tensor.
        tp_size: Tensor parallel size (for shape validation).
    
    Returns:
        All-reduced tensor.
    """
    if tp_size <= 1:
        return x
    return lax.psum(x, axis_name="tp")


def get_local_slice(
    global_array: jax.Array,
    tp_rank: int,
    tp_size: int,
    dim: int = 0,
) -> jax.Array:
    """Get the local slice of a globally sharded array.
    
    Args:
        global_array: The full array.
        tp_rank: This device's rank.
        tp_size: Total number of devices.
        dim: Dimension to slice.
    
    Returns:
        The local slice for this rank.
    """
    if tp_size <= 1:
        return global_array
    
    size = global_array.shape[dim]
    shard_size = size // tp_size
    start = tp_rank * shard_size
    
    # Use dynamic_slice for JAX compatibility
    start_indices = [0] * global_array.ndim
    start_indices[dim] = start
    
    slice_sizes = list(global_array.shape)
    slice_sizes[dim] = shard_size
    
    return lax.dynamic_slice(global_array, start_indices, slice_sizes)


# =============================================================================
# Weight Loading with Sharding
# =============================================================================

def load_sharded_weight(
    full_weight: jax.Array,
    mesh: Mesh,
    shard_type: str = "column",
) -> jax.Array:
    """Load and shard a weight matrix according to TP pattern.
    
    Args:
        full_weight: Complete weight matrix.
        mesh: Device mesh.
        shard_type: "column" (dim 0), "row" (dim 1), or "replicated".
    
    Returns:
        Properly sharded weight.
    """
    if mesh is None:
        return full_weight
    
    if shard_type == "column":
        return shard_weight_column(full_weight, mesh)
    elif shard_type == "row":
        return shard_weight_row(full_weight, mesh)
    else:
        return replicate(full_weight, mesh)


def create_sharded_zeros(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    mesh: Mesh,
    spec: P,
) -> jax.Array:
    """Create a sharded zero array.
    
    Args:
        shape: Global shape of the array.
        dtype: Data type.
        mesh: Device mesh.
        spec: Partition spec for sharding.
    
    Returns:
        Sharded zero array.
    """
    sharding = NamedSharding(mesh, spec)
    return jax.device_put(jnp.zeros(shape, dtype=dtype), sharding)
