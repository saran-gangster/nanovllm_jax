"""Weight loader for HuggingFace safetensors models.

Loads weights from safetensors files into Flax NNX models, handling:
- Fused QKV projections
- Fused gate/up projections
- Tensor parallelism sharding
- Weight name mapping between HuggingFace and our model
"""

import os
from glob import glob
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from flax import nnx

from nanovllm_jax.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from nanovllm_jax.layers.layernorm import RMSNorm
from nanovllm_jax.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


def tensor_to_jax(tensor) -> jax.Array:
    """Convert safetensors tensor to JAX array in bfloat16 for efficiency.
    
    Args:
        tensor: Tensor from safetensors (numpy-compatible).
    
    Returns:
        JAX array with same data, cast to bfloat16 for efficient inference.
    """
    # safetensors returns numpy arrays when using framework="np"
    # or torch tensors when using framework="pt"
    if hasattr(tensor, 'numpy'):
        # It's a torch tensor
        arr = jnp.array(tensor.numpy())
    else:
        # It's already numpy
        arr = jnp.array(tensor)
    
    # Cast to bfloat16 for efficient GPU inference (2x memory bandwidth)
    # Only cast floating point tensors
    if jnp.issubdtype(arr.dtype, jnp.floating):
        return arr.astype(jnp.bfloat16)
    return arr


def get_nested_attr(obj, path: str):
    """Get nested attribute by dot-separated path.
    
    Args:
        obj: Object to traverse.
        path: Dot-separated path like "model.layers.0.self_attn".
    
    Returns:
        The nested attribute.
    """
    parts = path.split(".")
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def parse_weight_name(weight_name: str) -> tuple[str, str]:
    """Parse HuggingFace weight name into module path and param name.
    
    Examples:
        "model.layers.0.self_attn.q_proj.weight" -> ("model.layers.0.self_attn.q_proj", "weight")
        "model.embed_tokens.weight" -> ("model.embed_tokens", "weight")
    
    Args:
        weight_name: Full weight name from safetensors.
    
    Returns:
        Tuple of (module_path, param_name).
    """
    parts = weight_name.rsplit(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def load_model(model: nnx.Module, path: str):
    """Load weights from safetensors into a Flax NNX model.
    
    Handles the packed_modules_mapping for fused layers:
    - q_proj, k_proj, v_proj -> qkv_proj
    - gate_proj, up_proj -> gate_up_proj
    
    Args:
        model: The Flax NNX model to load weights into.
        path: Path to directory containing safetensors files.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Find all safetensor files
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")
    
    loaded_count = 0
    skipped_count = 0
    
    for file in safetensor_files:
        with safe_open(file, framework="np", device="cpu") as f:
            for weight_name in f.keys():
                try:
                    loaded = _load_single_weight(
                        model, weight_name, f.get_tensor(weight_name),
                        packed_modules_mapping
                    )
                    if loaded:
                        loaded_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {weight_name}: {e}")
                    skipped_count += 1
    
    print(f"Loaded {loaded_count} weights, skipped {skipped_count}")


def _load_single_weight(
    model: nnx.Module,
    weight_name: str,
    tensor: np.ndarray,
    packed_modules_mapping: dict,
) -> bool:
    """Load a single weight tensor into the model.
    
    Args:
        model: The model to load into.
        weight_name: Full weight name from safetensors.
        tensor: The weight tensor to load.
        packed_modules_mapping: Mapping for fused layers.
    
    Returns:
        True if weight was loaded, False if skipped.
    """
    # Convert to JAX array
    jax_tensor = tensor_to_jax(tensor)
    
    # Check if this weight should be mapped to a packed module
    for orig_name, (packed_name, shard_id) in packed_modules_mapping.items():
        if orig_name in weight_name:
            # Replace original name with packed name
            module_path = weight_name.replace(f".{orig_name}.", f".{packed_name}.")
            module_path, param_name = parse_weight_name(module_path)
            
            try:
                module = get_nested_attr(model, module_path)
                _load_packed_weight(module, jax_tensor, shard_id, param_name)
                return True
            except (AttributeError, IndexError) as e:
                # Module doesn't exist in our model
                return False
    
    # Standard weight loading
    module_path, param_name = parse_weight_name(weight_name)
    
    try:
        module = get_nested_attr(model, module_path)
        _load_standard_weight(module, jax_tensor, param_name)
        return True
    except (AttributeError, IndexError):
        # Module doesn't exist in our model
        return False


def _load_packed_weight(
    module: nnx.Module,
    tensor: jax.Array,
    shard_id: str | int,
    param_name: str,
):
    """Load weight into a packed (fused) module.
    
    Args:
        module: The packed module (QKVParallelLinear or MergedColumnParallelLinear).
        tensor: Weight tensor to load.
        shard_id: Which shard ("q", "k", "v" for QKV, or 0/1 for gate/up).
        param_name: "weight" or "bias".
    """
    if isinstance(module, QKVParallelLinear):
        if param_name == "weight":
            module.load_weights(tensor, shard_id)
        elif param_name == "bias":
            module.load_weights(tensor, shard_id, loaded_bias=tensor)
    elif isinstance(module, MergedColumnParallelLinear):
        if param_name == "weight":
            module.load_weights(tensor, shard_id)
        elif param_name == "bias":
            module.load_weights(tensor, shard_id, loaded_bias=tensor)
    else:
        raise ValueError(f"Unknown packed module type: {type(module)}")


def _load_standard_weight(
    module: nnx.Module,
    tensor: jax.Array,
    param_name: str,
):
    """Load weight into a standard (non-packed) module.
    
    Args:
        module: The module to load into.
        tensor: Weight tensor to load.
        param_name: "weight" or "bias".
    """
    if isinstance(module, (ColumnParallelLinear, RowParallelLinear, ReplicatedLinear)):
        if param_name == "weight":
            module.load_weights(tensor)
        elif param_name == "bias":
            module.load_weights(tensor, loaded_bias=tensor)
    elif isinstance(module, (VocabParallelEmbedding, ParallelLMHead)):
        if param_name == "weight":
            module.load_weights(tensor)
    elif isinstance(module, RMSNorm):
        if param_name == "weight":
            module.load_weights(tensor)
    elif hasattr(module, param_name):
        # Generic fallback: try to set the attribute directly
        attr = getattr(module, param_name)
        if isinstance(attr, nnx.Param):
            attr.value = tensor
        elif isinstance(attr, nnx.Variable):
            attr.value = tensor
        else:
            setattr(module, param_name, tensor)
    else:
        raise AttributeError(f"Module {type(module)} has no attribute {param_name}")


def load_model_sharded(
    model: nnx.Module,
    path: str,
    tp_rank: int,
    tp_size: int,
):
    """Load weights with tensor parallelism awareness.
    
    This is a convenience wrapper that ensures the model was created
    with the correct tp_rank and tp_size, then loads weights.
    
    Args:
        model: Model created with correct tp_rank and tp_size.
        path: Path to safetensors files.
        tp_rank: This device's tensor parallel rank.
        tp_size: Total tensor parallel size.
    """
    # Verify model was created with correct TP settings
    if hasattr(model, 'tp_rank'):
        assert model.tp_rank == tp_rank, f"Model tp_rank {model.tp_rank} != {tp_rank}"
    if hasattr(model, 'tp_size'):
        assert model.tp_size == tp_size, f"Model tp_size {model.tp_size} != {tp_size}"
    
    # Load weights (sharding is handled by individual load_weights methods)
    load_model(model, path)
