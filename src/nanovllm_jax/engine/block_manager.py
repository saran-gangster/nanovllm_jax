"""Block manager for paged KV-cache with prefix caching.

Manages physical memory blocks for the KV-cache, handling:
- Block allocation and deallocation
- Prefix caching using content-based hashing
- Reference counting for shared blocks
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm_jax.engine.sequence import Sequence


class Block:
    """A single KV-cache block.
    
    Attributes:
        block_id: Unique physical block ID.
        ref_count: Number of sequences using this block.
        hash: Content hash for prefix caching (-1 if incomplete).
        token_ids: Cached token content for hash verification.
    """
    
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []
    
    def update(self, hash_value: int, token_ids: list[int]):
        """Update block with content hash and tokens.
        
        Called when a block becomes full (block_size tokens).
        
        Args:
            hash_value: xxhash of block content + prefix chain.
            token_ids: Tokens stored in this block.
        """
        self.hash = hash_value
        self.token_ids = token_ids
    
    def reset(self):
        """Reset block for reuse."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Manages paged KV-cache blocks with prefix caching.
    
    Handles block allocation, deallocation, and prefix cache lookups.
    Uses xxhash for fast content-based hashing with hash chains for
    prefix matching.
    
    Attributes:
        block_size: Number of tokens per block.
        blocks: List of all blocks.
        hash_to_block_id: Maps content hashes to block IDs.
        free_block_ids: Queue of available block IDs.
        used_block_ids: Set of currently allocated block IDs.
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        """Initialize block manager.
        
        Args:
            num_blocks: Total number of blocks available.
            block_size: Tokens per block.
        """
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # O(1) membership for free blocks; deque may contain stale entries.
        self.free_block_ids_set: set[int] = set(range(num_blocks))
        self._num_free_blocks = num_blocks
        self.used_block_ids: set[int] = set()
    
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """Compute content hash for prefix caching.
        
        Uses xxhash with hash chaining: hash(current_block) depends on
        hash(previous_block), creating a unique fingerprint for each
        prefix sequence.
        
        Args:
            token_ids: Tokens in the current block.
            prefix: Hash of the previous block (-1 if first block).
        
        Returns:
            64-bit hash value.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.fromiter(token_ids, dtype=np.int32, count=len(token_ids)).tobytes())
        return h.intdigest()

    def _allocate_next_free_block(self) -> Block:
        """Allocate the next free block from the queue head (common path)."""
        if not self.free_block_ids_set:
            raise RuntimeError("No free blocks available")

        while True:
            # Deque can be temporarily exhausted if it held only stale IDs.
            if not self.free_block_ids:
                self.free_block_ids.extend(self.free_block_ids_set)

            block_id = self.free_block_ids.popleft()
            if block_id not in self.free_block_ids_set:
                continue

            self.free_block_ids_set.remove(block_id)
            self._num_free_blocks -= 1
            block = self.blocks[block_id]
            if block.ref_count != 0:
                raise RuntimeError(f"Block {block_id} is already in use")
            block.reset()
            self.used_block_ids.add(block_id)
            return block
    
    def _allocate_block(self, block_id: int) -> Block:
        """Allocate a specific block.
        
        Args:
            block_id: ID of block to allocate.
        
        Returns:
            The allocated block.
        """
        block = self.blocks[block_id]
        if block_id not in self.free_block_ids_set:
            raise RuntimeError(f"Block {block_id} is not free")
        if block.ref_count != 0:
            raise RuntimeError(f"Block {block_id} is already in use")
        self.free_block_ids_set.remove(block_id)
        self._num_free_blocks -= 1
        block.reset()
        self.used_block_ids.add(block_id)
        return block
    
    def _deallocate_block(self, block_id: int):
        """Return a block to the free pool.
        
        Args:
            block_id: ID of block to free.
        """
        if self.blocks[block_id].ref_count != 0:
            raise RuntimeError(f"Block {block_id} still has a non-zero ref_count")
        self.used_block_ids.remove(block_id)
        if block_id in self.free_block_ids_set:
            return
        self.free_block_ids_set.add(block_id)
        self._num_free_blocks += 1
        self.free_block_ids.append(block_id)
    
    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we can allocate all blocks for a new sequence.
        
        Args:
            seq: Sequence needing block allocation.
        
        Returns:
            True if enough free blocks available.
        """
        return self._num_free_blocks >= seq.num_blocks
    
    def allocate(self, seq: Sequence):
        """Allocate blocks for a new sequence with prefix caching.
        
        Attempts to reuse cached blocks when content matches.
        Sets seq.num_cached_tokens to indicate cache hits.
        
        Args:
            seq: Sequence to allocate blocks for.
        """
        if seq.block_table:
            raise RuntimeError("Sequence already has blocks allocated")
        
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            
            # Only compute hash for full blocks
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # Try to find cached block
            block_id = self.hash_to_block_id.get(h, -1)
            
            # Verify content matches (collision protection)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # Allocate new block
                block = self._allocate_next_free_block()
                block_id = block.block_id
            else:
                # Reuse cached block
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            
            # Update hash table for full blocks
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)
    
    def deallocate(self, seq: Sequence):
        """Free all blocks for a sequence.
        
        Handles reference counting for shared blocks.
        
        Args:
            seq: Sequence to deallocate.
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        seq.num_cached_tokens = 0
        seq.block_table.clear()
    
    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append a new token (may need new block).
        
        A new block is needed when crossing a block boundary.
        
        Args:
            seq: Sequence to check.
        
        Returns:
            True if we can append.
        """
        # Need new block when at block boundary (token % block_size == 1 after append).
        if seq.num_tokens % self.block_size != 1:
            return True
        return self._num_free_blocks > 0

    def may_append(self, seq: Sequence) -> bool:
        """Handle block allocation/update after appending a token.

        - If crossing block boundary: allocate new block
        - If block just became full: compute hash and register

        Args:
            seq: Sequence that just had a token appended.

        Returns:
            True if block table changed (new block allocated), else False.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        mod = seq.num_tokens % self.block_size

        if mod == 1:
            # Just crossed into new block, need to allocate
            if last_block.hash == -1:
                raise RuntimeError("Previous block should already be complete")
            block = self._allocate_next_free_block()
            block_table.append(block.block_id)
            return True
        if mod == 0:
            # Block just became full, compute hash
            if last_block.hash != -1:
                raise RuntimeError("Incomplete block should not already have a hash")
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
            return False

        # Mid-block, nothing to do.
        if last_block.hash != -1:
            raise RuntimeError("Incomplete block should not already have a hash")
        return False
    
    @property
    def num_free_blocks(self) -> int:
        """Number of available blocks."""
        return self._num_free_blocks
    
    @property
    def num_used_blocks(self) -> int:
        """Number of allocated blocks."""
        return len(self.used_block_ids)
