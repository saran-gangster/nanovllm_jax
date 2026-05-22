"""Decode-schedule ownership for paged attention runtime state.

This module keeps decode schedule data as explicit runtime-owned state rather
than relying on module-global attention caches keyed by transient array ids.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import count
from threading import Lock
from typing import Any

import jax
import numpy as np


_TOKEN_COUNTER = count(1)
_PACKET_REGISTRY_LOCK = Lock()
_PACKET_REGISTRY: dict[int, "DecodeSchedulePacket"] = {}


def _array_fingerprint(
    host_array: np.ndarray | None,
    device_array: jax.Array | None,
    *,
    row_count: int | None = None,
) -> tuple[str, Any] | None:
    """Build a stable fingerprint for schedule data without device transfers."""
    if host_array is not None:
        view = host_array
        if row_count is not None:
            view = view[:row_count]
        contiguous = np.ascontiguousarray(view)
        digest = hashlib.blake2b(
            contiguous.view(np.uint8),
            digest_size=16,
        ).hexdigest()
        return ("host", digest)
    if device_array is not None:
        return ("device", id(device_array))
    return None


@dataclass
class DecodeScheduleHostView:
    """Host-side arrays associated with the active decode schedule."""

    input_ids: np.ndarray | None = None
    positions: np.ndarray | None = None
    slot_mapping: np.ndarray | None = None
    context_lens: np.ndarray | None = None
    block_tables: np.ndarray | None = None


@dataclass
class DecodeScheduleDeviceView:
    """Device-side arrays associated with the active decode schedule."""

    input_ids: jax.Array | None = None
    positions: jax.Array | None = None
    slot_mapping: jax.Array | None = None
    context_lens: jax.Array | None = None
    block_tables: jax.Array | None = None


@dataclass
class DecodeSchedulePacket:
    """Runner-owned decode schedule packet for one decode step/bucket.

    The packet owns the padded decode inputs and any prepared family-specific
    metadata reused across layers within a single trace/build of the decode
    graph. Prepared metadata is retained only when the refreshed schedule
    state is identical to the previous state; otherwise it is invalidated.
    """

    token: int
    real_batch_size: int = 0
    padded_batch_size: int = 0
    block_size: int = 0
    block_tables_dirty: bool = True
    sequence_ids: tuple[int, ...] = ()
    same_membership: bool = False
    last_decode_input_action: str | None = None
    last_block_table_action: str | None = None
    last_block_table_rows_changed: int = 0
    last_prepared_metadata_action: str | None = None
    last_prepared_metadata_entries_before: int = 0
    last_prepared_metadata_entries_after: int = 0
    refresh_generation: int = 0
    host: DecodeScheduleHostView = field(default_factory=DecodeScheduleHostView)
    device: DecodeScheduleDeviceView = field(default_factory=DecodeScheduleDeviceView)
    prepared_metadata_by_family: dict[str, dict[tuple[Any, ...], object]] = field(
        default_factory=dict
    )
    prepared_metadata_state_key: tuple[Any, ...] | None = None

    @property
    def block_tables(self) -> jax.Array | None:
        return self.device.block_tables

    @property
    def context_lens(self) -> jax.Array | None:
        return self.device.context_lens

    @property
    def slot_mapping(self) -> jax.Array | None:
        return self.device.slot_mapping

    @property
    def prepared_metadata_entries(self) -> int:
        return sum(len(cache) for cache in self.prepared_metadata_by_family.values())

    def _build_prepared_metadata_state_key(
        self,
        *,
        real_batch_size: int,
        padded_batch_size: int,
        block_size: int,
        sequence_ids: tuple[int, ...],
        host_view: DecodeScheduleHostView,
        device_view: DecodeScheduleDeviceView,
    ) -> tuple[Any, ...]:
        return (
            int(real_batch_size),
            int(padded_batch_size),
            int(block_size),
            tuple(sequence_ids),
            _array_fingerprint(
                host_view.context_lens,
                device_view.context_lens,
                row_count=padded_batch_size,
            ),
            _array_fingerprint(
                host_view.block_tables,
                device_view.block_tables,
                row_count=padded_batch_size,
            ),
        )

    def refresh(
        self,
        *,
        real_batch_size: int,
        padded_batch_size: int,
        block_size: int,
        block_tables_dirty: bool,
        sequence_ids: tuple[int, ...],
        same_membership: bool,
        decode_input_action: str,
        block_table_action: str,
        block_table_rows_changed: int,
        host_view: DecodeScheduleHostView,
        device_view: DecodeScheduleDeviceView,
    ) -> str:
        """Refresh packet inputs for the current decode step.

        Returns a short action label for diagnostics.
        """
        action = "reuse_block_tables"
        if self.device.block_tables is None:
            action = "create"
        elif self.padded_batch_size != padded_batch_size:
            action = "rebucket"
        elif not same_membership:
            action = "membership_changed"
        elif block_table_action == "full_block_table_transfer":
            action = "rebuild_block_tables"
        elif block_table_action == "patch_block_table_rows":
            action = "replace_block_tables"
        elif self.device.block_tables is not device_view.block_tables:
            action = "replace_block_tables"

        prepared_metadata_state_key = self._build_prepared_metadata_state_key(
            real_batch_size=real_batch_size,
            padded_batch_size=padded_batch_size,
            block_size=block_size,
            sequence_ids=sequence_ids,
            host_view=host_view,
            device_view=device_view,
        )
        prepared_metadata_entries_before = self.prepared_metadata_entries
        if self.prepared_metadata_state_key is None:
            prepared_metadata_action = "clear_initial"
            self.prepared_metadata_by_family.clear()
        elif self.prepared_metadata_state_key != prepared_metadata_state_key:
            prepared_metadata_action = "clear_schedule_changed"
            self.prepared_metadata_by_family.clear()
        else:
            prepared_metadata_action = "retain"

        self.real_batch_size = real_batch_size
        self.padded_batch_size = padded_batch_size
        self.block_size = block_size
        self.block_tables_dirty = block_tables_dirty
        self.sequence_ids = sequence_ids
        self.same_membership = same_membership
        self.last_decode_input_action = decode_input_action
        self.last_block_table_action = block_table_action
        self.last_block_table_rows_changed = block_table_rows_changed
        self.last_prepared_metadata_action = prepared_metadata_action
        self.last_prepared_metadata_entries_before = prepared_metadata_entries_before
        self.prepared_metadata_state_key = prepared_metadata_state_key
        self.host = host_view
        self.device = device_view
        self.refresh_generation += 1
        self.last_prepared_metadata_entries_after = self.prepared_metadata_entries
        return action

    def metadata_cache_for_family(self, family: str) -> dict[tuple[Any, ...], object]:
        return self.prepared_metadata_by_family.setdefault(str(family), {})

    def get_or_create_metadata(
        self,
        family: str,
        key: tuple[Any, ...],
        factory,
    ) -> object:
        family_cache = self.metadata_cache_for_family(family)
        metadata = family_cache.get(key)
        if metadata is None:
            metadata = factory()
            family_cache[key] = metadata
        return metadata


def allocate_decode_schedule_token() -> int:
    return next(_TOKEN_COUNTER)


def register_decode_schedule_packet(packet: DecodeSchedulePacket) -> None:
    with _PACKET_REGISTRY_LOCK:
        _PACKET_REGISTRY[packet.token] = packet


def get_decode_schedule_packet(token: int | None) -> DecodeSchedulePacket | None:
    if not token:
        return None
    with _PACKET_REGISTRY_LOCK:
        return _PACKET_REGISTRY.get(token)


def unregister_decode_schedule_packet(token: int | None) -> None:
    if not token:
        return
    with _PACKET_REGISTRY_LOCK:
        _PACKET_REGISTRY.pop(token, None)
