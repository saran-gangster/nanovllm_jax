"""Request scheduler for batching prefill and decode.

Handles scheduling of sequences for generation, managing:
- Prefill batching (processing prompts)
- Decode batching (generating tokens)
- Preemption when memory is tight
- Block allocation coordination with BlockManager
"""

from collections import deque

from nanovllm_jax.config import Config
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus
from nanovllm_jax.engine.block_manager import BlockManager


class Scheduler:
    """Schedules sequences for prefill and decode phases.

    Implements a two-phase scheduling strategy:
    1. Prefill: Process new prompts (priority over decode)
    2. Decode: Generate tokens for running sequences

    Uses preemption when memory is insufficient for decode.

    Attributes:
        max_num_seqs: Maximum sequences in a batch.
        max_num_batched_tokens: Maximum tokens in a prefill batch.
        eos: End-of-sequence token ID.
        block_manager: Manages KV-cache blocks.
        waiting: Queue of sequences waiting for prefill.
        running: Queue of sequences in decode phase.
    """

    def __init__(self, config: Config):
        """Initialize scheduler from config.

        Args:
            config: Engine configuration.
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # Fast-path cache: reuse last decode schedule when batch unchanged.
        self._cached_decode_seqs: list[Sequence] | None = None
        self._block_tables_dirty: bool = True
    
    def is_finished(self) -> bool:
        """Check if all sequences are complete.
        
        Returns:
            True if no sequences waiting or running.
        """
        return not self.waiting and not self.running
    
    def add(self, seq: Sequence):
        """Add a new sequence to the waiting queue.
        
        Args:
            seq: Sequence to add.
        """
        self.waiting.append(seq)
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule the next batch of sequences.

        Prioritizes prefill over decode. Handles preemption if
        memory is insufficient for decode.

        Returns:
            Tuple of (scheduled_sequences, is_prefill).
        """
        # Try prefill first (has priority)
        scheduled_seqs: list[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # Check if we can fit this sequence
            new_token_count = len(seq) - seq.num_cached_tokens
            if (num_batched_tokens + new_token_count > self.max_num_batched_tokens or
                not self.block_manager.can_allocate(seq)):
                break

            # Allocate blocks and schedule
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += new_token_count
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self._cached_decode_seqs = None  # Invalidate decode cache
            return scheduled_seqs, True  # is_prefill = True

        # ---- Decode fast path ----
        # If the batch is unchanged since last decode step (common case),
        # skip the deque pop/re-push and reuse the cached sequence list.
        if self._cached_decode_seqs is not None and not self.waiting:
            seqs = self._cached_decode_seqs
            block_table_changed = False
            for seq in seqs:
                if not self.block_manager.can_append(seq):
                    # Preemption needed; fall through to full path.
                    self._cached_decode_seqs = None
                    break
                if self.block_manager.may_append(seq):
                    block_table_changed = True
            else:
                if block_table_changed:
                    self._block_tables_dirty = True
                return seqs, False

        # ---- Decode full path ----
        self._cached_decode_seqs = None
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # Try to allocate block for new token
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Preempt last sequence to free memory
                    self.preempt(self.running.pop())
                else:
                    # Must preempt current sequence
                    self.preempt(seq)
                    break
            else:
                # Successfully allocated
                num_seqs += 1
                if self.block_manager.may_append(seq):
                    self._block_tables_dirty = True
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            raise RuntimeError(
                "Scheduler could not produce a decode batch. "
                "This usually indicates an internal block-management inconsistency."
            )

        # Re-add scheduled sequences to running queue
        self.running.extendleft(reversed(scheduled_seqs))
        self._cached_decode_seqs = scheduled_seqs
        self._block_tables_dirty = True
        return scheduled_seqs, False  # is_prefill = False
    
    def preempt(self, seq: Sequence):
        """Preempt a sequence to free memory.
        
        Moves sequence back to waiting queue and frees its blocks.
        
        Args:
            seq: Sequence to preempt.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    
    def postprocess(self, seqs: list[Sequence], token_ids):
        """Process newly generated tokens.
        
        Appends tokens to sequences and checks for completion.
        Sequences that hit EOS or max_tokens are finished and deallocated.
        
        Args:
            seqs: Sequences that generated tokens.
            token_ids: Generated token IDs (one per sequence). Can be a Python list
                or a NumPy/JAX host array.
        """
        finished_seq_ids: set[int] = set()
        eos = self.eos
        max_model_len = self.max_model_len
        for seq, token_id in zip(seqs, token_ids):
            # Inline the tiny append path to avoid per-token method overhead.
            seq.token_ids.append(token_id)
            seq.last_token = token_id
            seq.num_tokens += 1

            # Check completion conditions
            hit_eos = not seq.ignore_eos and token_id == eos
            hit_max_tokens = (seq.num_tokens - seq.num_prompt_tokens) >= seq.max_tokens
            hit_max_model_len = seq.num_tokens >= max_model_len
            
            if hit_eos or hit_max_tokens or hit_max_model_len:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                finished_seq_ids.add(seq.seq_id)

        # Avoid O(k * n) repeated deque.remove() when many sequences finish
        # in the same step; rebuild once while preserving order.
        if finished_seq_ids:
            self.running = deque(
                seq for seq in self.running if seq.seq_id not in finished_seq_ids
            )
            self._cached_decode_seqs = None  # Batch changed
            self._block_tables_dirty = True
    
    @property
    def num_waiting(self) -> int:
        """Number of sequences waiting for prefill."""
        return len(self.waiting)
    
    @property
    def num_running(self) -> int:
        """Number of sequences in decode phase."""
        return len(self.running)
