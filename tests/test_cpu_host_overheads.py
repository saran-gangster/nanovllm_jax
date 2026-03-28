"""Regression tests for host-side CPU overhead optimizations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.sampling_params import SamplingParams


def test_scheduler_decode_fast_path_keeps_block_tables_clean_when_unchanged() -> None:
    cfg = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        eos=-1,
        num_kvcache_blocks=128,
        kvcache_block_size=256,
    )
    scheduler = Scheduler(cfg)

    seq = Sequence([1] * 1024, SamplingParams(temperature=0.8, max_tokens=1024))
    scheduler.add(seq)

    # Prefill: allocate prompt blocks and produce first token.
    batch, is_prefill = scheduler.schedule()
    assert is_prefill
    scheduler.postprocess(batch, [2])  # seq len: 1025, should trigger block-table growth next decode

    # First decode comes from full path and marks block tables dirty.
    batch, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert scheduler._block_tables_dirty is True

    # Simulate LLMEngine.step consuming the dirty flag and appending one token.
    scheduler._block_tables_dirty = False
    scheduler.postprocess(batch, [2])  # seq len: 1026 (no new block boundary)

    # Next decode should stay clean when block tables are unchanged.
    batch2, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert scheduler._cached_decode_seqs == batch2
    assert scheduler._block_tables_dirty is False


def test_generate_uses_batch_decode_not_per_sequence_decode() -> None:
    class DummyTokenizer:
        def __init__(self):
            self.batch_decode_calls = 0
            self.decode_calls = 0

        def __call__(self, prompts, add_special_tokens=True, return_attention_mask=False):
            assert add_special_tokens is True
            assert return_attention_mask is False
            return {"input_ids": [[10 + i] for i in range(len(prompts))]}

        def batch_decode(self, token_ids, skip_special_tokens=True):
            self.batch_decode_calls += 1
            assert skip_special_tokens is True
            return [f"text_{i}" for i in range(len(token_ids))]

        def decode(self, *_args, **_kwargs):
            self.decode_calls += 1
            raise AssertionError("per-sequence decode should not be called")

    engine = LLMEngine.__new__(LLMEngine)
    engine.tokenizer = DummyTokenizer()

    added_prompts: list[list[int]] = []
    done = {"value": False}

    def add_request(prompt, _sampling_params):
        added_prompts.append(prompt)

    def is_finished():
        return done["value"]

    def step():
        done["value"] = True
        # Intentionally out of order to validate seq-id sorting.
        return [(1, [42, 43]), (0, [99])], -2

    engine.add_request = add_request
    engine.is_finished = is_finished
    engine.step = step

    outputs = engine.generate(
        prompts=["hello", "world"],
        sampling_params=SamplingParams(temperature=0.9, max_tokens=8),
        use_tqdm=False,
    )

    assert added_prompts == [[10], [11]]
    assert engine.tokenizer.batch_decode_calls == 1
    assert engine.tokenizer.decode_calls == 0
    assert outputs == [
        {"text": "text_0", "token_ids": [99]},
        {"text": "text_1", "token_ids": [42, 43]},
    ]


def test_prepare_decode_resets_padded_tail_when_batch_shrinks() -> None:
    sp = SamplingParams(temperature=0.8, max_tokens=64)

    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(max_num_seqs=16, decode_batch_min_size=8)
    runner.block_size = 256
    runner.max_blocks_per_seq = 16
    runner._decode_runtime_cache = None

    def make_seq(block_base: int) -> Sequence:
        seq = Sequence([1] * 1024, sp)
        seq.token_ids.extend([2] * 5)
        seq.num_tokens = len(seq.token_ids)
        seq.last_token = seq.token_ids[-1]
        seq.block_table = [block_base + i for i in range(seq.num_blocks)]
        return seq

    seqs8 = [make_seq(i * 8) for i in range(8)]
    seqs6 = [make_seq(i * 8) for i in range(6)]

    # First call seeds the full bucket with real values.
    runner._prepare_decode(seqs8, block_tables_dirty=True)
    input_ids, positions, context = runner._prepare_decode(seqs6, block_tables_dirty=False)

    input_ids_np = np.asarray(input_ids)
    positions_np = np.asarray(positions)
    slot_mapping_np = np.asarray(context.slot_mapping)
    context_lens_np = np.asarray(context.context_lens)

    # decode_batch_min_size=8 keeps both calls in same bucket; tail rows (6,7)
    # must be reset to safe dummy values.
    assert np.all(input_ids_np[6:] == 0)
    assert np.all(positions_np[6:] == 0)
    assert np.all(slot_mapping_np[6:] == -1)
    assert np.all(context_lens_np[6:] == 1)


def test_prepare_prefill_reuses_numpy_buffers_for_same_bucket() -> None:
    sp = SamplingParams(temperature=0.8, max_tokens=64)

    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(max_num_seqs=16, decode_batch_min_size=1, max_model_len=4096)
    runner.block_size = 256
    runner.max_blocks_per_seq = 16
    runner._prefill_runtime_cache = None
    runner._prefill_pos_template = None
    runner._prefill_slot_template = None

    def make_seq(block_base: int, prompt_len: int, cached_tokens: int) -> Sequence:
        seq = Sequence(list(range(prompt_len)), sp)
        seq.block_table = [block_base + i for i in range(seq.num_blocks)]
        seq.num_cached_tokens = cached_tokens
        return seq

    seqs = [
        make_seq(0, 800, 256),
        make_seq(8, 640, 0),
        make_seq(16, 420, 256),
        make_seq(24, 260, 0),
    ]

    runner._prepare_prefill(seqs)
    cache_first = runner._prefill_runtime_cache
    assert cache_first is not None
    first_ids = (
        id(cache_first.input_ids),
        id(cache_first.positions),
        id(cache_first.slot_mapping),
        id(cache_first.cu_seqlens_q),
        id(cache_first.cu_seqlens_k),
    )

    runner._prepare_prefill(seqs)
    cache_second = runner._prefill_runtime_cache
    assert cache_second is not None
    second_ids = (
        id(cache_second.input_ids),
        id(cache_second.positions),
        id(cache_second.slot_mapping),
        id(cache_second.cu_seqlens_q),
        id(cache_second.cu_seqlens_k),
    )

    assert first_ids == second_ids


def test_block_manager_cached_reuse_keeps_free_pool_consistent() -> None:
    sp = SamplingParams(temperature=0.8, max_tokens=64)
    old_block_size = Sequence.block_size
    Sequence.block_size = 2
    try:
        bm = BlockManager(num_blocks=6, block_size=2)

        seq_a = Sequence([1, 2, 3, 4], sp)
        bm.allocate(seq_a)
        bm.deallocate(seq_a)
        assert bm.num_free_blocks == 6

        # Same content should take the cached-block reuse path.
        seq_b = Sequence([1, 2, 3, 4], sp)
        bm.allocate(seq_b)
        assert bm.num_free_blocks == 4

        old_blocks = set(seq_b.block_table)
        seq_b.append_token(9)  # Next decode step will cross a block boundary.
        assert bm.can_append(seq_b)
        assert bm.may_append(seq_b) is True
        assert seq_b.block_table[-1] not in old_blocks

        bm.deallocate(seq_b)
        assert bm.num_free_blocks == 6
    finally:
        Sequence.block_size = old_block_size
