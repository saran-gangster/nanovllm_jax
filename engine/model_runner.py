"""Model runner for JAX inference with tensor parallelism.

Handles model execution including:
- KV-cache allocation with proper sharding
- Input preparation for prefill/decode
- JIT compilation with batch size buckets
- Multi-device coordination via Mesh + shard_map
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
from functools import partial

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
            # Estimate: use small KV cache for Kaggle compatibility
            # Each block stores: 2 * block_size * num_kv_heads * head_dim * num_layers * 4 bytes (fp32)
            bytes_per_block = (
                2 * self.block_size * num_kv_heads * head_dim * num_layers * 4
            )
            target_memory = 256 * 1024 * 1024  # 256MB - very conservative for Kaggle
            num_blocks = max(4, target_memory // bytes_per_block)  # At least 4 blocks
        
        config.num_kvcache_blocks = num_blocks
        print(f"Allocating {num_blocks} KV cache blocks")
        
        # Allocate KV cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # Using float32 for universal GPU compatibility (bf16/fp16 not supported on all GPUs)
        kv_cache_shape = (2, num_layers, num_blocks, self.block_size, num_kv_heads, head_dim)
        
        if self.mesh is not None:
            # With TP, KV cache heads are sharded across devices
            # Shape per device: [2, layers, blocks, block_size, num_kv_heads/tp, head_dim]
            kv_sharding = NamedSharding(
                self.mesh, 
                P(None, None, None, None, "tp", None)  # Shard on kv_heads dimension
            )
            self.kv_cache = jax.device_put(
                jnp.zeros(kv_cache_shape, dtype=jnp.float32),
                kv_sharding
            )
        else:
            self.kv_cache = jnp.zeros(kv_cache_shape, dtype=jnp.float32)
        
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
        
        Args:
            seqs: Sequences to prepare block tables for.
        
        Returns:
            Block tables array of shape [batch_size, max_blocks].
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return jnp.array(block_tables, dtype=jnp.int32)
    
    def _prepare_prefill(
        self,
        seqs: list[Sequence],
    ) -> tuple[jnp.ndarray, jnp.ndarray, AttentionContext]:
        """Prepare inputs for prefill phase.
        
        Packs multiple variable-length sequences into single tensors.
        
        Args:
            seqs: Sequences to prefill.
        
        Returns:
            Tuple of (input_ids, positions, context).
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        
        for seq in seqs:
            seqlen = len(seq)
            
            # Input tokens (skip cached)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            # Sequence lengths
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # Slot mapping (for KV cache storage)
            if seq.block_table:  # Skip if warmup
                for i in range(seq.num_cached_blocks, seq.num_blocks):
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens
                    slot_mapping.extend(list(range(start, end)))
        
        # Prepare block tables for prefix caching
        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self._prepare_block_tables(seqs)
        
        # Convert to JAX arrays
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
        positions = jnp.array(positions, dtype=jnp.int32)
        cu_seqlens_q = jnp.array(cu_seqlens_q, dtype=jnp.int32)
        cu_seqlens_k = jnp.array(cu_seqlens_k, dtype=jnp.int32)
        slot_mapping = jnp.array(slot_mapping, dtype=jnp.int32)
        
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
        
        Args:
            seqs: Sequences to decode.
        
        Returns:
            Tuple of (input_ids, positions, context).
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
        positions = jnp.array(positions, dtype=jnp.int32)
        slot_mapping = jnp.array(slot_mapping, dtype=jnp.int32)
        context_lens = jnp.array(context_lens, dtype=jnp.int32)
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
