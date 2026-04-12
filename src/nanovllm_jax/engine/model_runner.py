"""Model runner for local JAX inference.

Handles model execution including:
- KV-cache allocation with proper sharding
- Input preparation for prefill/decode
- JIT compilation with batch size buckets
- internal backend runtime state for attention dispatch

Optimizations:
- Disables x64 mode to reduce memory bandwidth
- Uses contiguous memory layouts for better cache locality
"""

import logging
import os
import time
from dataclasses import dataclass
# Disable x64 mode BEFORE importing jax for reduced memory bandwidth
# This must be set before JAX is imported
os.environ.setdefault('JAX_ENABLE_X64', 'False')

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
from functools import partial
from typing import TYPE_CHECKING

# Verify x64 is disabled
if jax.config.jax_enable_x64:
    logger.warning("x64 mode is enabled; inference performance may be reduced")

from nanovllm_jax.config import Config
from nanovllm_jax.engine.decode_schedule import (
    DecodeScheduleDeviceView,
    DecodeScheduleHostView,
    DecodeSchedulePacket,
    allocate_decode_schedule_token,
    register_decode_schedule_packet,
    unregister_decode_schedule_packet,
)
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.layers.sampler import Sampler
from nanovllm_jax.layers.attention import configure_prefill_backend
from nanovllm_jax.layers.paged_attention import (
    configure_attention_backends,
    create_attention_backend_runtime_state,
)
from nanovllm_jax.utils.context import AttentionContext, create_prefill_context, create_decode_context
from nanovllm_jax.utils.parallel import create_tp_mesh
from nanovllm_jax.utils.runtime_diagnostics import (
    append_decode_schedule_record,
    block_until_ready_tree,
    consume_partitioned_decode_reduction_stats,
    consume_kv_update_stats,
    decode_schedule_dump_enabled,
    decode_step_profiling_enabled,
    reset_partitioned_decode_reduction_stats,
    reset_kv_update_stats,
)

if TYPE_CHECKING:  # pragma: no cover
    from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class PrefillRuntimeCache:
    batch_size: int
    token_capacity: int
    input_ids: np.ndarray
    positions: np.ndarray
    slot_mapping: np.ndarray
    cu_seqlens_q: np.ndarray
    cu_seqlens_k: np.ndarray


@dataclass
class DecodeRuntimeCache:
    batch_size: int
    input_ids: np.ndarray
    positions: np.ndarray
    slot_mapping: np.ndarray
    context_lens: np.ndarray
    input_ids_jax: jax.Array | None
    positions_jax: jax.Array | None
    slot_mapping_jax: jax.Array | None
    context_lens_jax: jax.Array | None
    block_tables_host: np.ndarray | None
    block_tables_jax: jax.Array | None
    schedule_packet: DecodeSchedulePacket | None
    sequence_ids: tuple[int, ...]


class ModelRunner:
    """Runs the model for inference.
    
    Handles:
    - Model loading and initialization
    - KV-cache allocation and management
    - Input preparation for prefill and decode phases
    - JIT compilation with batch size buckets
    - single-runtime attention backend ownership

    Attributes:
        config: Engine configuration.
        block_size: Tokens per KV-cache block.
        enforce_eager: If True, disable JIT compilation.
        tp_size: Internal tensor-parallel width. Public runtime support is `1` only.
        tp_rank: This device's tensor parallel rank.
        mesh: JAX device mesh for tensor-parallel internals.
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
            tp_rank: Tensor-parallel rank (always 0 in supported public mode).
        """
        self.config = config
        hf_config = config.hf_config

        # Apply Config-driven attention backend selection before anything else.
        self._attention_backend_runtime = create_attention_backend_runtime_state()
        configure_attention_backends(config, runtime_state=self._attention_backend_runtime)
        configure_prefill_backend(config)

        self.block_size = config.kvcache_block_size
        self._decode_schedule_token = allocate_decode_schedule_token()
        self._decode_runtime_cache: DecodeRuntimeCache | None = None
        self._prefill_runtime_cache: PrefillRuntimeCache | None = None
        self._last_decode_profile: dict | None = None
        self._last_decode_prepare_stats: dict | None = None
        self._last_decode_schedule_action: str | None = None
        self._prefill_pos_template = np.arange(
            max(1, config.max_model_len), dtype=np.int32,
        )
        self._prefill_slot_template = np.arange(self.block_size, dtype=np.int32)
        self.max_blocks_per_seq = (config.max_model_len + self.block_size - 1) // self.block_size
        self.enforce_eager = config.enforce_eager
        self.tp_size = config.tensor_parallel_size
        self.tp_rank = tp_rank
        
        # Create device mesh for tensor parallelism
        self.mesh = create_tp_mesh(self.tp_size) if self.tp_size > 1 else None
        
        # Initialize RNG
        self.rngs = nnx.Rngs(0)
        
        # Create model with mesh context
        logger.info("Creating model from config: %s", config.model)
        from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM

        self.model = Qwen3ForCausalLM(
            hf_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            block_size=self.block_size,
            mesh=self.mesh,
            rngs=self.rngs,
        )
        
        # Load weights
        logger.info("Loading weights from: %s", config.model)
        from nanovllm_jax.utils.loader import load_model

        load_model(self.model, config.model)
        
        # Create sampler
        self.sampler = Sampler(rngs=self.rngs)
        
        # JIT compile model functions BEFORE warmup
        # (warmup needs the compiled function)
        self._run_model_jit = None  # Initialize to None
        if not self.enforce_eager:
            logger.info("JIT compiling model")
            self._compile_model()
        
        # Allocate KV cache first (needed for warmup)
        logger.info("Allocating KV cache")
        self._allocate_kv_cache()
        
        # Warmup after everything is set up
        logger.info("Warming up model")
        self._warmup_model()
        
        logger.info("Model runner initialized")

    def _ensure_decode_schedule_token(self) -> int:
        token = getattr(self, "_decode_schedule_token", 0)
        if not token:
            token = allocate_decode_schedule_token()
            self._decode_schedule_token = token
        return token

    def _update_decode_input_device_arrays(
        self,
        cache: DecodeRuntimeCache,
        *,
        real_batch_size: int,
        batch_size: int,
        same_membership: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, str]:
        """Materialize decode input arrays with cheap device-side patching when possible."""
        if (
            cache.input_ids_jax is None
            or cache.positions_jax is None
            or cache.slot_mapping_jax is None
            or cache.context_lens_jax is None
        ):
            cache.input_ids_jax = jnp.asarray(cache.input_ids)
            cache.positions_jax = jnp.asarray(cache.positions)
            cache.slot_mapping_jax = jnp.asarray(cache.slot_mapping)
            cache.context_lens_jax = jnp.asarray(cache.context_lens)
            return (
                cache.input_ids_jax,
                cache.positions_jax,
                cache.slot_mapping_jax,
                cache.context_lens_jax,
                "full_transfer",
            )

        active_slice = slice(0, real_batch_size)
        input_ids = cache.input_ids_jax.at[active_slice].set(
            jnp.asarray(cache.input_ids[active_slice])
        )
        positions = cache.positions_jax.at[active_slice].set(
            jnp.asarray(cache.positions[active_slice])
        )
        slot_mapping = cache.slot_mapping_jax.at[active_slice].set(
            jnp.asarray(cache.slot_mapping[active_slice])
        )
        context_lens = cache.context_lens_jax.at[active_slice].set(
            jnp.asarray(cache.context_lens[active_slice])
        )

        if real_batch_size < batch_size:
            tail = slice(real_batch_size, batch_size)
            input_ids = input_ids.at[tail].set(0)
            positions = positions.at[tail].set(0)
            slot_mapping = slot_mapping.at[tail].set(-1)
            context_lens = context_lens.at[tail].set(1)

        cache.input_ids_jax = input_ids
        cache.positions_jax = positions
        cache.slot_mapping_jax = slot_mapping
        cache.context_lens_jax = context_lens
        action = "patch_active_rows" if same_membership else "patch_rows_membership_changed"
        return input_ids, positions, slot_mapping, context_lens, action

    def _update_decode_block_tables(
        self,
        cache: DecodeRuntimeCache,
        seqs: list[Sequence],
        *,
        batch_size: int,
        block_tables_dirty: bool,
    ) -> tuple[jax.Array, str, int]:
        """Update cached block tables with row-wise host/device patching when possible."""
        max_len = self.max_blocks_per_seq
        host = cache.block_tables_host
        full_rebuild_required = False
        if host is None or host.shape != (batch_size, max_len):
            host = np.full((batch_size, max_len), -1, dtype=np.int32)
            cache.block_tables_host = host
            cache.block_tables_jax = None
            full_rebuild_required = True

        if not block_tables_dirty and cache.block_tables_jax is not None:
            return cache.block_tables_jax, "reuse_block_tables", 0

        changed_rows: list[int] = []
        for row_idx, seq in enumerate(seqs):
            row = host[row_idx]
            row_len = len(seq.block_table)
            row_changed = (
                row_len > 0 and not np.array_equal(row[:row_len], np.asarray(seq.block_table, dtype=np.int32))
            ) or np.any(row[row_len:] != -1)
            if row_changed:
                row.fill(-1)
                if row_len:
                    row[:row_len] = seq.block_table
                changed_rows.append(row_idx)

        for row_idx in range(len(seqs), batch_size):
            row = host[row_idx]
            if np.any(row != -1):
                row.fill(-1)
                changed_rows.append(row_idx)

        if cache.block_tables_jax is None or full_rebuild_required:
            block_tables = jnp.asarray(host)
            cache.block_tables_jax = block_tables
            return block_tables, "full_block_table_transfer", len(changed_rows)

        if not changed_rows:
            return cache.block_tables_jax, "reuse_block_tables", 0

        changed_rows_np = host[changed_rows].copy()
        row_indices = jnp.asarray(np.asarray(changed_rows, dtype=np.int32))
        block_tables = cache.block_tables_jax.at[row_indices].set(
            jnp.asarray(changed_rows_np)
        )
        cache.block_tables_jax = block_tables
        return block_tables, "patch_block_table_rows", len(changed_rows)
    
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
        logger.info("Allocating %s KV cache blocks", num_blocks)
        
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
    
    def _prepare_block_tables(
        self,
        seqs: list[Sequence],
        batch_size: int | None = None,
    ) -> jnp.ndarray:
        """Prepare block tables tensor for attention.
        
        Uses NumPy for efficient CPU-side padding.
        
        Args:
            seqs: Sequences to prepare block tables for.
        
        Returns:
            Block tables array of shape [batch_size, max_blocks].
        """
        real_batch_size = len(seqs)
        batch_size = batch_size or real_batch_size
        max_len = self.max_blocks_per_seq
        
        # Pre-allocate with -1 padding
        block_tables = np.full((batch_size, max_len), -1, dtype=np.int32)
        
        for i, seq in enumerate(seqs):
            bt_len = len(seq.block_table)
            block_tables[i, :bt_len] = seq.block_table
        
        return jnp.asarray(block_tables)

    def _bucket_decode_batch_size(self, batch_size: int) -> int:
        """Bucket decode batch size to reduce JIT recompilations.

        Tiered strategy to balance recompilation count vs wasted compute:
        - <=64: power-of-2 (few distinct shapes, fine for small batch)
        - 65-256: round up to nearest 16 (tighter fit, avoids large gaps)
        - >256: round up to nearest 32 (still tight, fewer shapes than pow2)
        """
        max_bs = self.config.max_num_seqs
        min_bs = min(self.config.decode_batch_min_size, max_bs)
        target = max(batch_size, min_bs)
        if target <= 64:
            bucket = 1 << (target - 1).bit_length()
        elif target <= 256:
            bucket = ((target + 15) // 16) * 16
        else:
            bucket = ((target + 31) // 32) * 32
        return min(bucket, max_bs)

    def _bucket_prefill_batch_size(self, batch_size: int) -> int:
        """Bucket prefill batch size to reduce JIT recompilations."""
        max_bs = self.config.max_num_seqs
        target = max(batch_size, 1)
        bucket = 1 << (target - 1).bit_length()
        return min(bucket, max_bs)

    def _ensure_prefill_templates(self, min_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Ensure reusable NumPy templates exist for prefill packing."""
        pos_template = getattr(self, "_prefill_pos_template", None)
        slot_template = getattr(self, "_prefill_slot_template", None)

        if slot_template is None or slot_template.shape[0] != self.block_size:
            slot_template = np.arange(self.block_size, dtype=np.int32)
            self._prefill_slot_template = slot_template

        required = max(1, int(min_seq_len))
        if pos_template is None or pos_template.shape[0] < required:
            new_len = 1 << (required - 1).bit_length()
            pos_template = np.arange(new_len, dtype=np.int32)
            self._prefill_pos_template = pos_template

        return pos_template, slot_template
    
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
        real_batch_size = len(seqs)
        batch_size = self._bucket_prefill_batch_size(real_batch_size)
        total_q_tokens = sum(seq.num_tokens - seq.num_cached_tokens for seq in seqs)
        max_seq_len = max((seq.num_tokens for seq in seqs), default=1)

        # Reuse pre-allocated host buffers when bucket and capacities match.
        token_capacity = max(1, total_q_tokens)
        cache = self._prefill_runtime_cache
        if (
            cache is None
            or cache.batch_size != batch_size
            or cache.token_capacity < token_capacity
        ):
            cache = PrefillRuntimeCache(
                batch_size=batch_size,
                token_capacity=token_capacity,
                input_ids=np.empty(token_capacity, dtype=np.int32),
                positions=np.empty(token_capacity, dtype=np.int32),
                slot_mapping=np.empty(token_capacity, dtype=np.int32),
                cu_seqlens_q=np.zeros(batch_size + 1, dtype=np.int32),
                cu_seqlens_k=np.zeros(batch_size + 1, dtype=np.int32),
            )
            self._prefill_runtime_cache = cache

        input_ids = cache.input_ids
        positions = cache.positions
        slot_mapping = cache.slot_mapping
        cu_seqlens_q = cache.cu_seqlens_q
        cu_seqlens_k = cache.cu_seqlens_k
        cu_seqlens_q[0] = 0
        cu_seqlens_k[0] = 0

        pos_template, slot_template = self._ensure_prefill_templates(max_seq_len)
        
        max_seqlen_q = 0
        max_seqlen_k = 0
        token_idx = 0
        slot_idx = 0
        block_size = self.block_size
        
        for i, seq in enumerate(seqs):
            seqlen = seq.num_tokens
            num_cached = seq.num_cached_tokens
            seqlen_q = seqlen - num_cached
            seqlen_k = seqlen
            
            # Update cumulative lengths
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_q
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlen_k
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # Fill input tokens and positions using slices
            end = token_idx + seqlen_q
            input_ids[token_idx:end] = seq.token_ids[num_cached:seqlen]
            positions[token_idx:end] = pos_template[num_cached:seqlen]
            token_idx += seqlen_q
            
            # Slot mapping for uncached blocks.
            block_table = seq.block_table
            if block_table:
                num_cached_blocks = seq.num_cached_blocks
                num_blocks = seq.num_blocks
                # Full blocks use a reusable 0..block_size-1 template.
                for block_i in range(num_cached_blocks, max(num_cached_blocks, num_blocks - 1)):
                    start = block_table[block_i] * block_size
                    slot_mapping[slot_idx:slot_idx + block_size] = slot_template + start
                    slot_idx += block_size
                if num_cached_blocks < num_blocks:
                    block_len = seq.last_block_num_tokens
                    start = block_table[num_blocks - 1] * block_size
                    slot_mapping[slot_idx:slot_idx + block_len] = slot_template[:block_len] + start
                    slot_idx += block_len

        # Pad cumulative sequence lengths for bucketed (dummy) rows.
        if batch_size != real_batch_size:
            final_q = cu_seqlens_q[real_batch_size]
            final_k = cu_seqlens_k[real_batch_size]
            cu_seqlens_q[real_batch_size + 1:] = final_q
            cu_seqlens_k[real_batch_size + 1:] = final_k
        
        # Prepare block tables for prefix caching
        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self._prepare_block_tables(seqs, batch_size=batch_size)
        
        # Convert to JAX arrays (single transfer to GPU)
        input_ids = jnp.asarray(input_ids[:total_q_tokens])
        positions = jnp.asarray(positions[:total_q_tokens])
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
        block_tables_dirty: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, AttentionContext]:
        """Prepare inputs for decode phase.

        Each sequence contributes exactly one token (the last).
        Uses NumPy for efficient CPU-side preparation.

        Args:
            seqs: Sequences to decode.
            block_tables_dirty: If False, reuse cached block_tables JAX array.

        Returns:
            Tuple of (input_ids, positions, context).
        """
        real_batch_size = len(seqs)
        batch_size = self._bucket_decode_batch_size(real_batch_size)
        seq_ids = tuple(seq.seq_id for seq in seqs)

        # Reuse pre-allocated NumPy arrays when the bucket hasn't changed.
        cache = self._decode_runtime_cache
        if cache is None or cache.batch_size != batch_size:
            cache = DecodeRuntimeCache(
                batch_size=batch_size,
                input_ids=np.zeros(batch_size, dtype=np.int32),
                positions=np.zeros(batch_size, dtype=np.int32),
                slot_mapping=np.full(batch_size, -1, dtype=np.int32),
                context_lens=np.ones(batch_size, dtype=np.int32),
                input_ids_jax=None,
                positions_jax=None,
                slot_mapping_jax=None,
                context_lens_jax=None,
                block_tables_host=None,
                block_tables_jax=None,
                schedule_packet=None,
                sequence_ids=(),
            )
            self._decode_runtime_cache = cache
            block_tables_dirty = True

        np_input_ids = cache.input_ids
        np_positions = cache.positions
        np_slot_mapping = cache.slot_mapping
        np_context_lens = cache.context_lens

        if real_batch_size:
            bs = self.block_size
            np_input_ids[:real_batch_size] = np.fromiter(
                (seq.last_token for seq in seqs),
                dtype=np.int32,
                count=real_batch_size,
            )
            np_context_lens[:real_batch_size] = np.fromiter(
                (seq.num_tokens for seq in seqs),
                dtype=np.int32,
                count=real_batch_size,
            )
            np_positions[:real_batch_size] = np_context_lens[:real_batch_size] - 1
            np_slot_mapping[:real_batch_size] = np.fromiter(
                (
                    seq.block_table[-1] * bs + ((seq.num_tokens - 1) % bs)
                    for seq in seqs
                ),
                dtype=np.int32,
                count=real_batch_size,
            )

        # Keep padded rows stable when batch size shrinks within same bucket.
        if real_batch_size != batch_size:
            np_input_ids[real_batch_size:] = 0
            np_positions[real_batch_size:] = 0
            np_context_lens[real_batch_size:] = 1
            np_slot_mapping[real_batch_size:] = -1

        same_membership = seq_ids == cache.sequence_ids
        (
            input_ids,
            positions,
            slot_mapping,
            context_lens,
            decode_input_action,
        ) = self._update_decode_input_device_arrays(
            cache,
            real_batch_size=real_batch_size,
            batch_size=batch_size,
            same_membership=same_membership,
        )
        block_tables, block_table_action, block_table_rows_changed = (
            self._update_decode_block_tables(
                cache,
                seqs,
                batch_size=batch_size,
                block_tables_dirty=block_tables_dirty,
            )
        )
        cache.sequence_ids = seq_ids

        schedule_packet = cache.schedule_packet
        if schedule_packet is None:
            schedule_packet = DecodeSchedulePacket(
                token=self._ensure_decode_schedule_token(),
            )
            cache.schedule_packet = schedule_packet

        schedule_action = schedule_packet.refresh(
            real_batch_size=real_batch_size,
            padded_batch_size=batch_size,
            block_size=self.block_size,
            block_tables_dirty=block_tables_dirty,
            sequence_ids=seq_ids,
            same_membership=same_membership,
            decode_input_action=decode_input_action,
            block_table_action=block_table_action,
            block_table_rows_changed=block_table_rows_changed,
            host_view=DecodeScheduleHostView(
                input_ids=cache.input_ids,
                positions=cache.positions,
                slot_mapping=cache.slot_mapping,
                context_lens=cache.context_lens,
                block_tables=cache.block_tables_host,
            ),
            device_view=DecodeScheduleDeviceView(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            ),
        )
        register_decode_schedule_packet(schedule_packet)
        self._last_decode_schedule_action = schedule_action
        self._last_decode_prepare_stats = {
            "real_batch_size": real_batch_size,
            "padded_batch_size": batch_size,
            "sequence_membership_unchanged": same_membership,
            "decode_input_action": decode_input_action,
            "block_table_action": block_table_action,
            "block_table_rows_changed": block_table_rows_changed,
            "decode_schedule_action": schedule_action,
            "prepared_metadata_action": schedule_packet.last_prepared_metadata_action,
            "prepared_metadata_entries_before": (
                schedule_packet.last_prepared_metadata_entries_before
            ),
            "prepared_metadata_entries_after": (
                schedule_packet.last_prepared_metadata_entries_after
            ),
        }

        if decode_schedule_dump_enabled():
            append_decode_schedule_record(
                {
                    "event": "decode_schedule_refresh",
                    "action": schedule_action,
                    "real_batch_size": real_batch_size,
                    "padded_batch_size": batch_size,
                    "block_size": self.block_size,
                    "block_tables_dirty": bool(block_tables_dirty),
                    "decode_schedule_token": schedule_packet.token,
                    "sequence_membership_unchanged": same_membership,
                    "decode_input_action": decode_input_action,
                    "block_table_action": block_table_action,
                    "block_table_rows_changed": block_table_rows_changed,
                    "prepared_metadata_action": (
                        schedule_packet.last_prepared_metadata_action
                    ),
                    "prepared_metadata_entries_before": (
                        schedule_packet.last_prepared_metadata_entries_before
                    ),
                    "prepared_metadata_entries": schedule_packet.prepared_metadata_entries,
                }
            )

        context = create_decode_context(
            context_lens=context_lens,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            decode_schedule_token=schedule_packet.token,
        )

        return input_ids, positions, context
    
    def _prepare_sample(
        self,
        seqs: list[Sequence],
        padded_batch_size: int | None = None,
    ) -> jnp.ndarray:
        """Prepare sampling temperatures.
        
        Args:
            seqs: Sequences to sample for.
            padded_batch_size: Optional padded decode batch size.
        
        Returns:
            Temperature array of shape [batch_size].
        """
        real_batch_size = len(seqs)
        if padded_batch_size is None or padded_batch_size == real_batch_size:
            temperatures = np.fromiter(
                (seq.temperature for seq in seqs),
                dtype=np.float32,
                count=real_batch_size,
            )
            return jnp.asarray(temperatures)

        temperatures = np.ones((padded_batch_size,), dtype=np.float32)
        temperatures[:real_batch_size] = np.fromiter(
            (seq.temperature for seq in seqs),
            dtype=np.float32,
            count=real_batch_size,
        )
        return jnp.asarray(temperatures)
    
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
        block_tables_dirty: bool = True,
    ) -> np.ndarray | None:
        """Run inference for a batch of sequences.
        
        Args:
            seqs: Sequences to process.
            is_prefill: True for prefill phase, False for decode.
        
        Returns:
            List of sampled token IDs (only on rank 0 for TP).
        """
        real_batch_size = len(seqs)
        self._last_decode_profile = None
        self._last_decode_prepare_stats = None

        # Prepare inputs
        if is_prefill:
            input_ids, positions, context = self._prepare_prefill(seqs)
        else:
            profile_enabled = decode_step_profiling_enabled()
            prepare_started_at = time.perf_counter() if profile_enabled else 0.0
            input_ids, positions, context = self._prepare_decode(
                seqs, block_tables_dirty=block_tables_dirty,
            )
            if profile_enabled:
                self._last_decode_profile = {
                    "phase": "decode",
                    "real_batch_size": real_batch_size,
                    "padded_batch_size": int(input_ids.shape[0]),
                    "prepare_decode_s": time.perf_counter() - prepare_started_at,
                    "decode_schedule_action": self._last_decode_schedule_action,
                }
                if self._last_decode_prepare_stats is not None:
                    self._last_decode_profile.update(self._last_decode_prepare_stats)
                reset_kv_update_stats()
                reset_partitioned_decode_reduction_stats()
        profile_enabled = bool(self._last_decode_profile is not None)
        
        # Run model
        model_started_at = time.perf_counter() if profile_enabled else 0.0
        logits = self._run_model(input_ids, positions, context, is_prefill)
        if profile_enabled:
            block_until_ready_tree(logits)
            assert self._last_decode_profile is not None
            self._last_decode_profile["model_execute_s"] = (
                time.perf_counter() - model_started_at
            )

            kv_update_stats = consume_kv_update_stats()
            self._last_decode_profile["kv_update_s"] = (
                kv_update_stats["seconds"] if kv_update_stats["measured"] else None
            )
            self._last_decode_profile["kv_update_calls"] = kv_update_stats["calls"]
            self._last_decode_profile["kv_update_tokens"] = kv_update_stats["tokens"]
            self._last_decode_profile["kv_update_valid_tokens"] = (
                kv_update_stats["valid_tokens"]
            )
            self._last_decode_profile["kv_update_skipped_tokens"] = (
                kv_update_stats["skipped_tokens"]
            )
            self._last_decode_profile["kv_update_duplicate_slots"] = (
                kv_update_stats["duplicate_slots"]
            )
            self._last_decode_profile["kv_update_backend"] = (
                kv_update_stats["backend"]
                or os.environ.get("NANOVLLM_JAX_KV_UPDATE_BACKEND", "scatter")
                .strip()
                .lower()
            )
            self._last_decode_profile["kv_update_measured"] = bool(
                kv_update_stats["measured"]
            )
            reduction_stats = consume_partitioned_decode_reduction_stats()
            self._last_decode_profile["partitioned_decode_reduction_s"] = (
                reduction_stats["seconds"] if reduction_stats["measured"] else None
            )
            self._last_decode_profile["partitioned_decode_reduction_calls"] = (
                reduction_stats["calls"]
            )
            self._last_decode_profile["partitioned_decode_reduction_backend"] = (
                reduction_stats["backend"]
            )
            self._last_decode_profile["partitioned_decode_reduction_family"] = (
                reduction_stats["family"]
            )
            self._last_decode_profile["partitioned_decode_reduction_max_splits"] = (
                reduction_stats["max_splits"]
            )
            self._last_decode_profile["partitioned_decode_reduction_measured"] = bool(
                reduction_stats["measured"]
            )

        # Prepare sampling (pad to bucketed batch size when needed).
        temperatures = (
            self._prepare_sample(seqs, padded_batch_size=logits.shape[0])
            if self.tp_rank == 0
            else None
        )
        
        # Sample tokens (only on rank 0)
        if self.tp_rank == 0 and logits is not None:
            sample_started_at = time.perf_counter() if profile_enabled else 0.0
            token_ids = self.sampler(logits, temperatures)
            if profile_enabled:
                block_until_ready_tree(token_ids)
                assert self._last_decode_profile is not None
                self._last_decode_profile["sampler_s"] = (
                    time.perf_counter() - sample_started_at
                )
            # Slice away padded rows.
            if token_ids.shape[0] != real_batch_size:
                token_ids = token_ids[:real_batch_size]
            # Move to host as a compact NumPy array (avoids slow Python `.tolist()`).
            return jax.device_get(token_ids)
        
        return None
    
    def exit(self):
        """Cleanup resources."""
        # JAX handles cleanup automatically, but we can explicitly delete
        # large arrays if needed
        unregister_decode_schedule_packet(getattr(self, "_decode_schedule_token", 0))
        cache = getattr(self, "_decode_runtime_cache", None)
        if cache is not None:
            cache.schedule_packet = None
        self._decode_schedule_token = 0
        self._last_decode_schedule_action = None
        self._last_decode_prepare_stats = None
        if hasattr(self, "kv_cache"):
            del self.kv_cache
        if hasattr(self, "model"):
            del self.model

    def consume_last_decode_profile(self) -> dict | None:
        profile = self._last_decode_profile
        self._last_decode_profile = None
        return profile
