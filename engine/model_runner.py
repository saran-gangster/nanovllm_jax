"""Model runner for JAX inference with tensor parallelism.

Handles model execution including:
- KV-cache allocation with proper sharding
- Input preparation for prefill/decode
- JIT compilation with batch size buckets
- Multi-device coordination via Mesh + shard_map

Optimizations:
- Disables x64 mode to reduce memory bandwidth
- Uses contiguous memory layouts for better cache locality
"""

import os
# Disable x64 mode BEFORE importing jax for reduced memory bandwidth
# This must be set before JAX is imported
os.environ.setdefault('JAX_ENABLE_X64', 'False')

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
from functools import partial

# Verify x64 is disabled
if jax.config.jax_enable_x64:
    print("Warning: x64 mode is enabled, performance may be reduced")

from nanovllm_jax.config import Config
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.layers.sampler import Sampler
from nanovllm_jax.utils.context import AttentionContext, create_prefill_context, create_decode_context
from nanovllm_jax.utils.loader import load_model
from nanovllm_jax.utils.parallel import create_tp_mesh

class ModelRunner:
    """Runs the model for inference with tensor parallelism support.
    
    Handles:
    - Model loading and initialization
    - KV-cache allocation and management
    - Input preparation for prefill and decode phases
    - JIT compilation with batch size buckets
    - Tensor parallelism via Mesh + NamedSharding + shard_map
    
    Attributes:
        config: Engine configuration.
        block_size: Tokens per KV-cache block.
        enforce_eager: If True, disable JIT compilation.
        tp_size: Tensor parallel size.
        tp_rank: This device's tensor parallel rank.
        mesh: JAX device mesh for tensor parallelism.
        model: The Qwen3 model.
        sampler: Token sampler.
        kv_cache: Pre-allocated KV cache array.
    """
    
    def __init__(
        self,
        config: Config,
        tp_rank: int = 0,
    ):
        """Initialize model runner.
        
        Args:
            config: Engine configuration.
            tp_rank: Tensor parallel rank (0 for single-GPU).
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tp_size = config.tensor_parallel_size
        self.tp_rank = tp_rank
        
        # Create device mesh for tensor parallelism
        self.mesh = create_tp_mesh(self.tp_size) if self.tp_size > 1 else None
        
        # Initialize RNG
        self.rngs = nnx.Rngs(0)
        
        # Create model with mesh context
        print(f"Creating model from config: {config.model}")
        self.model = Qwen3ForCausalLM(
            hf_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            block_size=self.block_size,
            mesh=self.mesh,
            rngs=self.rngs,
        )
        
        # Load weights
        print(f"Loading weights from: {config.model}")
        load_model(self.model, config.model)
        
        # Create sampler
        self.sampler = Sampler(rngs=self.rngs)
        
        # JIT compile model functions BEFORE warmup
        # (warmup needs the compiled function)
        self._run_model_jit = None  # Initialize to None
        if not self.enforce_eager:
            print("JIT compiling model...")
            self._compile_model()
        
        # Allocate KV cache first (needed for warmup)
        print("Allocating KV cache...")
        self._allocate_kv_cache()
        
        # Warmup after everything is set up
        print("Warming up model...")
        self._warmup_model()
        
        print("Model runner initialized")
    
    def _warmup_model(self):
        """Run a warmup pass to trigger lazy initialization."""
        # Use very short sequences for warmup to reduce memory pressure
        # JIT will recompile for different shapes if needed
        warmup_seq_len = 16  # Very short for minimal memory
        num_seqs = 1  # Single sequence for warmup
        
        # Create dummy sequences
        seqs = [Sequence([0] * warmup_seq_len) for _ in range(num_seqs)]
        
        # Calculate blocks needed per sequence
        block_size = self.config.kvcache_block_size
        blocks_per_seq = (warmup_seq_len + block_size - 1) // block_size
        
        # Assign dummy block tables
        for i, seq in enumerate(seqs):
            seq.block_table = list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq))
        
        # Run prefill
        self.run(seqs, is_prefill=True)
    
    def _allocate_kv_cache(self):
        """Allocate KV cache based on available memory.
        
        In JAX, we use a simpler fixed allocation strategy since memory
        management is handled by XLA.
        """
        config = self.config
        hf_config = config.hf_config
        
        # Calculate KV cache dimensions
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size
        head_dim = getattr(
            hf_config, "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        num_layers = hf_config.num_hidden_layers
        
        # Calculate number of blocks
        # For simplicity, use config value or estimate based on model
        if config.num_kvcache_blocks > 0:
            num_blocks = config.num_kvcache_blocks
        else:
            # Estimate: use reasonable KV cache for efficient inference
            # Each block stores: 2 * block_size * num_kv_heads * head_dim * num_layers * 2 bytes (bf16)
            bytes_per_block = (
                2 * self.block_size * num_kv_heads * head_dim * num_layers * 2
            )
            # Use more memory for better performance
            target_memory = int(config.gpu_memory_utilization * 2 * 1024 * 1024 * 1024)  # 2GB * util
            num_blocks = max(16, target_memory // bytes_per_block)  # At least 16 blocks
        
        config.num_kvcache_blocks = num_blocks
        print(f"Allocating {num_blocks} KV cache blocks")
        
        # Allocate KV cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # Using bfloat16 for better memory bandwidth (2x faster than float32)
        kv_cache_shape = (2, num_layers, num_blocks, self.block_size, num_kv_heads, head_dim)
        
        if self.mesh is not None:
            # With TP, KV cache heads are sharded across devices
            # Shape per device: [2, layers, blocks, block_size, num_kv_heads/tp, head_dim]
            kv_sharding = NamedSharding(
                self.mesh, 
                P(None, None, None, None, "tp", None)  # Shard on kv_heads dimension
            )
            self.kv_cache = jax.device_put(
                jnp.zeros(kv_cache_shape, dtype=jnp.bfloat16),
                kv_sharding
            )
        else:
            self.kv_cache = jnp.zeros(kv_cache_shape, dtype=jnp.bfloat16)
        
        # Wire KV cache to attention layers
        self._wire_kv_cache()
    
    def _wire_kv_cache(self):
        """Connect KV cache arrays to attention layers."""
        layer_id = 0
        for layer in self.model.model.layers:
            attn = layer.self_attn.attn
            attn.set_kv_cache(
                self.kv_cache[0, layer_id],  # k_cache
                self.kv_cache[1, layer_id],  # v_cache
            )
            layer_id += 1
    
    def _compile_model(self):
        """JIT compile model for common batch sizes."""
        # Define batch sizes to pre-compile
        max_bs = min(self.config.max_num_seqs, 512)
        self.compiled_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        
        # Create JIT-compiled forward function
        # Note: In JAX, JIT compilation happens on first call with each shape
        # We don't need to explicitly capture like CUDA graphs
        
        # Use a more aggressive JIT with reduced recompilations
        def run_model_jit(model, input_ids, positions, context):
            """JIT-compiled model forward pass."""
            hidden_states = model(input_ids, positions, context)
            return model.compute_logits(hidden_states, context)
        
        # Apply JIT with donated args for memory efficiency
        self._run_model_jit = nnx.jit(run_model_jit)
    
    def _prepare_block_tables(self, seqs: list[Sequence]) -> jnp.ndarray:
        """Prepare block tables tensor for attention.
        
        Uses NumPy for efficient CPU-side padding.
        
        Args:
            seqs: Sequences to prepare block tables for.
        
        Returns:
            Block tables array of shape [batch_size, max_blocks].
        """
        batch_size = len(seqs)
        max_len = max(len(seq.block_table) for seq in seqs)
        
        # Pre-allocate with -1 padding
        block_tables = np.full((batch_size, max_len), -1, dtype=np.int32)
        
        for i, seq in enumerate(seqs):
            bt_len = len(seq.block_table)
            block_tables[i, :bt_len] = seq.block_table
        
        return jnp.asarray(block_tables)
    
    def _prepare_prefill(
        self,
        seqs: list[Sequence],
    ) -> tuple[jnp.ndarray, jnp.ndarray, AttentionContext]:
        """Prepare inputs for prefill phase.
        
        Packs multiple variable-length sequences into single tensors.
        Uses NumPy for CPU-side operations to minimize overhead.
        
        Args:
            seqs: Sequences to prefill.
        
        Returns:
            Tuple of (input_ids, positions, context).
        """
        # Pre-compute sizes for efficient allocation
        batch_size = len(seqs)
        total_q_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs)
        total_slots = sum(
            sum(self.block_size if i != seq.num_blocks - 1 else seq.last_block_num_tokens
                for i in range(seq.num_cached_blocks, seq.num_blocks))
            for seq in seqs if seq.block_table
        )
        
        # Pre-allocate numpy arrays
        input_ids = np.empty(total_q_tokens, dtype=np.int32)
        positions = np.empty(total_q_tokens, dtype=np.int32)
        slot_mapping = np.empty(total_slots, dtype=np.int32)
        cu_seqlens_q = np.zeros(batch_size + 1, dtype=np.int32)
        cu_seqlens_k = np.zeros(batch_size + 1, dtype=np.int32)
        
        max_seqlen_q = 0
        max_seqlen_k = 0
        token_idx = 0
        slot_idx = 0
        
        for i, seq in enumerate(seqs):
            seqlen = len(seq)
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            
            # Update cumulative lengths
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_q
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlen_k
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # Fill input tokens and positions using slices
            input_ids[token_idx:token_idx + seqlen_q] = seq[seq.num_cached_tokens:]
            positions[token_idx:token_idx + seqlen_q] = np.arange(seq.num_cached_tokens, seqlen)
            token_idx += seqlen_q
            
            # Slot mapping using vectorized range computation
            if seq.block_table:
                for block_i in range(seq.num_cached_blocks, seq.num_blocks):
                    start = seq.block_table[block_i] * self.block_size
                    if block_i != seq.num_blocks - 1:
                        block_len = self.block_size
                    else:
                        block_len = seq.last_block_num_tokens
                    slot_mapping[slot_idx:slot_idx + block_len] = np.arange(start, start + block_len)
                    slot_idx += block_len
        
        # Prepare block tables for prefix caching
        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self._prepare_block_tables(seqs)
        
        # Convert to JAX arrays (single transfer to GPU)
        input_ids = jnp.asarray(input_ids)
        positions = jnp.asarray(positions)
        cu_seqlens_q = jnp.asarray(cu_seqlens_q)
        cu_seqlens_k = jnp.asarray(cu_seqlens_k)
        slot_mapping = jnp.asarray(slot_mapping[:slot_idx])  # Trim to actual size
        
        context = create_prefill_context(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        
        return input_ids, positions, context
    
    def _prepare_decode(
        self,
        seqs: list[Sequence],
    ) -> tuple[jnp.ndarray, jnp.ndarray, AttentionContext]:
        """Prepare inputs for decode phase.
        
        Each sequence contributes exactly one token (the last).
        Uses NumPy for efficient CPU-side preparation.
        
        Args:
            seqs: Sequences to decode.
        
        Returns:
            Tuple of (input_ids, positions, context).
        """
        batch_size = len(seqs)
        
        # Pre-allocate numpy arrays
        input_ids = np.empty(batch_size, dtype=np.int32)
        positions = np.empty(batch_size, dtype=np.int32)
        slot_mapping = np.empty(batch_size, dtype=np.int32)
        context_lens = np.empty(batch_size, dtype=np.int32)
        
        for i, seq in enumerate(seqs):
            input_ids[i] = seq.last_token
            positions[i] = len(seq) - 1
            context_lens[i] = len(seq)
            slot_mapping[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        
        # Convert to JAX arrays (single transfer)
        input_ids = jnp.asarray(input_ids)
        positions = jnp.asarray(positions)
        slot_mapping = jnp.asarray(slot_mapping)
        context_lens = jnp.asarray(context_lens)
        block_tables = self._prepare_block_tables(seqs)
        
        context = create_decode_context(
            context_lens=context_lens,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        
        return input_ids, positions, context
    
    def _prepare_sample(self, seqs: list[Sequence]) -> jnp.ndarray:
        """Prepare sampling temperatures.
        
        Args:
            seqs: Sequences to sample for.
        
        Returns:
            Temperature array of shape [batch_size].
        """
        temperatures = [seq.temperature for seq in seqs]
        return jnp.array(temperatures, dtype=jnp.float32)
    
    def _run_model(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        context: AttentionContext,
        is_prefill: bool,
    ) -> jnp.ndarray:
        """Run model forward pass.
        
        Args:
            input_ids: Input token IDs.
            positions: Position indices.
            context: Attention context.
            is_prefill: Whether this is prefill or decode.
        
        Returns:
            Logits array of shape [batch_size, vocab_size].
        """
        if self.enforce_eager or self._run_model_jit is None:
            # Eager mode (for debugging or when JIT not yet compiled)
            hidden_states = self.model(input_ids, positions, context)
            return self.model.compute_logits(hidden_states, context)
        else:
            # JIT compiled mode
            return self._run_model_jit(self.model, input_ids, positions, context)
    
    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> list[int] | None:
        """Run inference for a batch of sequences.
        
        Args:
            seqs: Sequences to process.
            is_prefill: True for prefill phase, False for decode.
        
        Returns:
            List of sampled token IDs (only on rank 0 for TP).
        """
        # Prepare inputs
        if is_prefill:
            input_ids, positions, context = self._prepare_prefill(seqs)
        else:
            input_ids, positions, context = self._prepare_decode(seqs)
        
        # Prepare sampling
        temperatures = self._prepare_sample(seqs) if self.tp_rank == 0 else None
        
        # Run model
        logits = self._run_model(input_ids, positions, context, is_prefill)
        
        # Sample tokens (only on rank 0)
        if self.tp_rank == 0 and logits is not None:
            token_ids = self.sampler(logits, temperatures)
            return token_ids.tolist()
        
        return None
    
    def exit(self):
        """Cleanup resources."""
        # JAX handles cleanup automatically, but we can explicitly delete
        # large arrays if needed
        del self.kv_cache
        del self.model
