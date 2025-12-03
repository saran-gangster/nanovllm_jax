"""Sequence state management for LLM inference.

Tracks per-request state including tokens, block allocation, and generation status.
This is pure Python host-side logic (no JAX tensors).
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm_jax.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Status of a sequence in the generation pipeline."""
    WAITING = auto()   # In queue, blocks not yet allocated
    RUNNING = auto()   # Actively generating, blocks allocated
    FINISHED = auto()  # Generation complete (hit EOS or max_tokens)


class Sequence:
    """Represents a single generation request.
    
    Tracks all state for one sequence including:
    - Token history (prompt + generated)
    - Block allocation for KV-cache
    - Sampling parameters
    - Generation status
    
    Attributes:
        block_size: Class-level constant for KV-cache block size.
        counter: Class-level counter for unique sequence IDs.
    """
    
    block_size: int = 256
    counter = count()
    
    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = None):
        """Initialize a new sequence.
        
        Args:
            token_ids: Prompt token IDs.
            sampling_params: Generation parameters (temperature, max_tokens, etc.).
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0  # Tokens from prefix cache hit
        self.block_table: list[int] = []  # Physical block IDs
        
        # Sampling parameters
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
    
    def __len__(self) -> int:
        """Number of tokens (prompt + completion)."""
        return self.num_tokens
    
    def __getitem__(self, key):
        """Index into token_ids."""
        return self.token_ids[key]
    
    @property
    def is_finished(self) -> bool:
        """Check if generation is complete."""
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self) -> int:
        """Number of generated tokens."""
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def prompt_token_ids(self) -> list[int]:
        """Original prompt tokens."""
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_token_ids(self) -> list[int]:
        """Generated tokens."""
        return self.token_ids[self.num_prompt_tokens:]
    
    @property
    def num_cached_blocks(self) -> int:
        """Number of blocks covered by prefix cache."""
        return self.num_cached_tokens // self.block_size
    
    @property
    def num_blocks(self) -> int:
        """Total number of blocks needed for current token count."""
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self) -> int:
        """Number of tokens in the last (potentially partial) block."""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    def block(self, i: int) -> list[int]:
        """Get token IDs for the i-th block.
        
        Args:
            i: Block index (0-indexed).
        
        Returns:
            Token IDs for that block.
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]
    
    def append_token(self, token_id: int):
        """Append a newly generated token.
        
        Args:
            token_id: The new token to append.
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def __getstate__(self):
        """Custom pickle state for efficient multi-process communication.
        
        Only serializes essential state, using last_token instead of full
        token_ids when possible.
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )
    
    def __setstate__(self, state):
        """Restore from pickled state."""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
    
    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.name}, "
            f"tokens={self.num_tokens}, blocks={len(self.block_table)})"
        )
