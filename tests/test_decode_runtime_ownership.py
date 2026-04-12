"""Tests for decode-schedule ownership and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.engine.decode_schedule import (
    DecodeScheduleDeviceView,
    DecodeScheduleHostView,
    DecodeSchedulePacket,
    get_decode_schedule_packet,
    unregister_decode_schedule_packet,
)
from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.sampling_params import SamplingParams
from nanovllm_jax.utils.context import create_decode_context


def _make_runner() -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(max_num_seqs=16, decode_batch_min_size=8)
    runner.block_size = 256
    runner.max_blocks_per_seq = 16
    runner._decode_runtime_cache = None
    runner._decode_schedule_token = 0
    runner._last_decode_schedule_action = None
    runner._last_decode_profile = None
    return runner


def _make_seq(block_base: int, extra_tokens: int = 0) -> Sequence:
    sp = SamplingParams(temperature=0.8, max_tokens=64)
    seq = Sequence([1] * 1024, sp)
    if extra_tokens:
        seq.token_ids.extend([2] * extra_tokens)
        seq.num_tokens = len(seq.token_ids)
        seq.last_token = seq.token_ids[-1]
    seq.block_table = [block_base + i for i in range(seq.num_blocks)]
    return seq


def _refresh_direct_packet(
    packet: DecodeSchedulePacket,
    *,
    context_lens: np.ndarray,
    block_tables: np.ndarray,
    sequence_ids: tuple[int, ...] = (10, 11),
    padded_batch_size: int = 2,
    block_size: int = 256,
) -> None:
    packet.refresh(
        real_batch_size=len(sequence_ids),
        padded_batch_size=padded_batch_size,
        block_size=block_size,
        block_tables_dirty=True,
        sequence_ids=sequence_ids,
        same_membership=True,
        decode_input_action="full_transfer",
        block_table_action="full_block_table_transfer",
        block_table_rows_changed=len(sequence_ids),
        host_view=DecodeScheduleHostView(
            context_lens=context_lens,
            block_tables=block_tables,
        ),
        device_view=DecodeScheduleDeviceView(
            context_lens=jnp.asarray(context_lens),
            block_tables=jnp.asarray(block_tables),
        ),
    )


def test_prepare_decode_registers_schedule_packet() -> None:
    runner = _make_runner()
    seqs = [_make_seq(i * 8) for i in range(8)]

    _input_ids, _positions, context = runner._prepare_decode(seqs, block_tables_dirty=True)

    assert context.decode_schedule_token != 0
    packet = get_decode_schedule_packet(context.decode_schedule_token)
    assert packet is not None
    assert packet.block_tables is context.block_tables
    assert packet.context_lens is context.context_lens
    assert packet.slot_mapping is context.slot_mapping
    assert packet.device.input_ids is not None
    assert packet.device.positions is not None
    assert packet.host.block_tables is not None
    assert packet.host.slot_mapping is not None
    assert packet.real_batch_size == len(seqs)
    assert packet.padded_batch_size == 8

    unregister_decode_schedule_packet(context.decode_schedule_token)


def test_decode_schedule_packet_retains_metadata_for_identical_host_content() -> None:
    packet = DecodeSchedulePacket(token=1)
    context_lens = np.asarray([128, 256], dtype=np.int32)
    block_tables = np.asarray([[1, 2], [3, 4]], dtype=np.int32)

    _refresh_direct_packet(packet, context_lens=context_lens, block_tables=block_tables)
    packet.get_or_create_metadata("baseline", ("baseline", 2), object)

    _refresh_direct_packet(
        packet,
        context_lens=context_lens.copy(),
        block_tables=block_tables.copy(),
    )

    assert packet.last_prepared_metadata_action == "retain"
    assert packet.last_prepared_metadata_entries_before == 1
    assert packet.last_prepared_metadata_entries_after == 1
    assert packet.prepared_metadata_entries == 1


def test_decode_schedule_packet_clears_metadata_when_sequence_ids_change() -> None:
    packet = DecodeSchedulePacket(token=1)
    context_lens = np.asarray([128, 256], dtype=np.int32)
    block_tables = np.asarray([[1, 2], [3, 4]], dtype=np.int32)

    _refresh_direct_packet(packet, context_lens=context_lens, block_tables=block_tables)
    packet.get_or_create_metadata("throughput", ("throughput", 2), object)

    _refresh_direct_packet(
        packet,
        context_lens=context_lens.copy(),
        block_tables=block_tables.copy(),
        sequence_ids=(10, 12),
    )

    assert packet.last_prepared_metadata_action == "clear_schedule_changed"
    assert packet.last_prepared_metadata_entries_before == 1
    assert packet.last_prepared_metadata_entries_after == 0
    assert packet.prepared_metadata_entries == 0


def test_prepare_decode_reuses_schedule_packet_for_same_bucket() -> None:
    runner = _make_runner()
    seqs = [_make_seq(i * 8) for i in range(8)]

    _input_ids1, _positions1, context1 = runner._prepare_decode(seqs, block_tables_dirty=True)
    packet1 = get_decode_schedule_packet(context1.decode_schedule_token)
    assert packet1 is not None
    packet1.get_or_create_metadata("baseline", ("baseline", 8), object)
    packet1.get_or_create_metadata("latency", ("latency", 8), object)
    packet1.get_or_create_metadata("throughput", ("throughput", 8), object)

    _input_ids2, _positions2, context2 = runner._prepare_decode(seqs, block_tables_dirty=False)
    packet2 = get_decode_schedule_packet(context2.decode_schedule_token)
    assert packet2 is packet1
    assert context2.decode_schedule_token == context1.decode_schedule_token
    assert packet2.prepared_metadata_entries == 3
    assert packet2.sequence_ids == tuple(seq.seq_id for seq in seqs)
    assert packet2.same_membership is True
    assert packet2.last_decode_input_action == "patch_active_rows"
    assert packet2.last_block_table_action == "reuse_block_tables"
    assert packet2.last_prepared_metadata_action == "retain"
    assert packet2.last_prepared_metadata_entries_before == 3
    assert packet2.last_prepared_metadata_entries_after == 3
    assert runner._last_decode_schedule_action == "reuse_block_tables"
    assert runner._last_decode_prepare_stats is not None
    assert runner._last_decode_prepare_stats["sequence_membership_unchanged"] is True
    assert runner._last_decode_prepare_stats["decode_input_action"] == "patch_active_rows"
    assert runner._last_decode_prepare_stats["block_table_action"] == "reuse_block_tables"
    assert runner._last_decode_prepare_stats["block_table_rows_changed"] == 0
    assert runner._last_decode_prepare_stats["prepared_metadata_action"] == "retain"
    assert runner._last_decode_prepare_stats["prepared_metadata_entries_before"] == 3
    assert runner._last_decode_prepare_stats["prepared_metadata_entries_after"] == 3

    unregister_decode_schedule_packet(context2.decode_schedule_token)


def test_prepare_decode_patches_only_changed_block_table_rows() -> None:
    runner = _make_runner()
    seqs = [_make_seq(i * 8) for i in range(8)]

    _input_ids1, _positions1, context1 = runner._prepare_decode(seqs, block_tables_dirty=True)
    original_block_tables = context1.block_tables
    packet1 = get_decode_schedule_packet(context1.decode_schedule_token)
    assert packet1 is not None
    packet1.get_or_create_metadata("baseline", ("baseline", 8), object)
    packet1.get_or_create_metadata("throughput", ("throughput", 8), object)

    seqs[0].append_token(7)
    seqs[0].block_table.append(999)

    _input_ids2, _positions2, context2 = runner._prepare_decode(seqs, block_tables_dirty=True)

    assert context2.decode_schedule_token == context1.decode_schedule_token
    assert context2.block_tables is not original_block_tables
    assert runner._last_decode_prepare_stats is not None
    assert runner._last_decode_prepare_stats["sequence_membership_unchanged"] is True
    assert runner._last_decode_prepare_stats["decode_input_action"] == "patch_active_rows"
    assert runner._last_decode_prepare_stats["block_table_action"] == "patch_block_table_rows"
    assert runner._last_decode_prepare_stats["block_table_rows_changed"] >= 1
    assert runner._last_decode_prepare_stats["prepared_metadata_action"] == (
        "clear_schedule_changed"
    )
    assert runner._last_decode_prepare_stats["prepared_metadata_entries_before"] == 2
    assert runner._last_decode_prepare_stats["prepared_metadata_entries_after"] == 0
    packet2 = get_decode_schedule_packet(context2.decode_schedule_token)
    assert packet2 is not None
    assert packet2.last_block_table_action == "patch_block_table_rows"
    assert packet2.last_prepared_metadata_action == "clear_schedule_changed"
    assert packet2.host.block_tables is not None
    assert packet2.device.block_tables is context2.block_tables
    assert packet2.prepared_metadata_entries == 0

    unregister_decode_schedule_packet(context2.decode_schedule_token)


def test_prepare_decode_dumps_schedule_diagnostics(monkeypatch, tmp_path: Path) -> None:
    runner = _make_runner()
    seqs = [_make_seq(i * 8) for i in range(8)]

    monkeypatch.setenv("NANOVLLM_JAX_DUMP_DECODE_SCHEDULE", "1")
    monkeypatch.setenv("NANOVLLM_JAX_DIAGNOSTICS_DIR", str(tmp_path))

    _input_ids, _positions, context = runner._prepare_decode(seqs, block_tables_dirty=True)

    payloads = [
        json.loads(line)
        for line in (tmp_path / "decode_schedule.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert payloads
    latest = payloads[-1]
    assert latest["event"] == "decode_schedule_refresh"
    assert latest["action"] == "create"
    assert latest["real_batch_size"] == 8
    assert latest["decode_schedule_token"] == context.decode_schedule_token
    assert latest["decode_input_action"] == "full_transfer"
    assert latest["block_table_action"] == "full_block_table_transfer"
    assert latest["prepared_metadata_action"] == "clear_initial"
    assert latest["prepared_metadata_entries_before"] == 0

    unregister_decode_schedule_packet(context.decode_schedule_token)


def test_model_runner_exit_unregisters_decode_schedule_packet() -> None:
    runner = _make_runner()
    seqs = [_make_seq(i * 8) for i in range(8)]

    _input_ids, _positions, context = runner._prepare_decode(seqs, block_tables_dirty=True)
    token = context.decode_schedule_token
    assert token != 0
    assert get_decode_schedule_packet(token) is not None
    assert runner._decode_runtime_cache is not None
    assert runner._decode_runtime_cache.schedule_packet is not None

    runner.exit()

    assert get_decode_schedule_packet(token) is None
    assert runner._decode_schedule_token == 0
    assert runner._last_decode_schedule_action is None
    assert runner._last_decode_prepare_stats is None
    assert runner._decode_runtime_cache is not None
    assert runner._decode_runtime_cache.schedule_packet is None


def test_prepare_decode_reallocates_token_after_exit_reset() -> None:
    runner = _make_runner()
    seqs = [_make_seq(i * 8) for i in range(8)]

    _input_ids, _positions, context = runner._prepare_decode(seqs, block_tables_dirty=True)
    token = context.decode_schedule_token
    assert token != 0

    runner.exit()

    replacement = runner._ensure_decode_schedule_token()
    assert replacement != 0
    assert replacement != token
    unregister_decode_schedule_packet(replacement)


def test_attention_context_pytree_roundtrip_preserves_decode_schedule_token() -> None:
    context = create_decode_context(
        context_lens=jnp.asarray([128, 256], dtype=jnp.int32),
        slot_mapping=jnp.asarray([7, -1], dtype=jnp.int32),
        block_tables=jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        decode_schedule_token=12345,
    )

    leaves, treedef = jax.tree_util.tree_flatten(context)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.decode_schedule_token == 12345
    assert rebuilt.is_prefill is False
    assert rebuilt.max_seqlen_q == 0
    assert rebuilt.max_seqlen_k == 0
    assert jnp.array_equal(rebuilt.context_lens, context.context_lens)
    assert jnp.array_equal(rebuilt.slot_mapping, context.slot_mapping)
    assert jnp.array_equal(rebuilt.block_tables, context.block_tables)


def test_llm_engine_step_writes_decode_step_profile(monkeypatch, tmp_path: Path) -> None:
    class DummyScheduler:
        def __init__(self):
            self._block_tables_dirty = True

        def schedule(self):
            seq = SimpleNamespace(seq_id=0, completion_token_ids=[42], is_finished=True)
            return [seq], False

        def postprocess(self, _seqs, _token_ids):
            return None

    class DummyRunner:
        def run(self, _seqs, _is_prefill, block_tables_dirty=True):
            assert block_tables_dirty is True
            return [7]

        def consume_last_decode_profile(self):
            return {
                "prepare_decode_s": 0.01,
                "model_execute_s": 0.02,
                "sampler_s": 0.003,
                "kv_update_s": None,
                "kv_update_calls": 0,
                "kv_update_tokens": 0,
                "kv_update_backend": "scatter",
                "kv_update_measured": False,
                "padded_batch_size": 1,
                "real_batch_size": 1,
                "decode_input_action": "patch_active_rows",
                "block_table_action": "reuse_block_tables",
                "block_table_rows_changed": 0,
                "sequence_membership_unchanged": True,
            }

    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = DummyScheduler()
    engine.model_runner = DummyRunner()

    monkeypatch.setenv("NANOVLLM_JAX_PROFILE_DECODE_STEP", "1")
    monkeypatch.setenv("NANOVLLM_JAX_DIAGNOSTICS_DIR", str(tmp_path))

    outputs, num_tokens = engine.step()

    assert outputs == [(0, [42])]
    assert num_tokens == -1

    payloads = [
        json.loads(line)
        for line in (tmp_path / "decode_step_profile.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert payloads
    latest = payloads[-1]
    assert latest["event"] == "decode_step_profile"
    assert latest["scheduled_batch_size"] == 1
    assert latest["block_tables_dirty"] is True
    assert "scheduler_s" in latest
    assert "postprocess_s" in latest
    assert latest["decode_input_action"] == "patch_active_rows"
