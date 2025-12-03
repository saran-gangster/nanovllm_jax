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
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
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
        scheduled_seqs = []
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
            return scheduled_seqs, True  # is_prefill = True
        
        # Decode: schedule running sequences
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
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs, "No sequences scheduled"
        
        # Re-add scheduled sequences to running queue
        self.running.extendleft(reversed(scheduled_seqs))
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
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """Process newly generated tokens.
        
        Appends tokens to sequences and checks for completion.
        Sequences that hit EOS or max_tokens are finished and deallocated.
        
        Args:
            seqs: Sequences that generated tokens.
            token_ids: Generated token IDs (one per sequence).
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            
            # Check completion conditions
            hit_eos = not seq.ignore_eos and token_id == self.eos
            hit_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            
            if hit_eos or hit_max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
    
    @property
    def num_waiting(self) -> int:
        """Number of sequences waiting for prefill."""
        return len(self.waiting)
    
    @property
    def num_running(self) -> int:
        """Number of sequences in decode phase."""
        return len(self.running)
