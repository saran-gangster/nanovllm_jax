"""Paged attention dispatch and fallback kernels.

This module owns the public paged-attention runtime surface:
- a dense vectorized reference path,
- a blockwise streaming fallback path,
- and Mosaic GPU dispatch for baseline, latency, throughput, and throughput-v2
  preview kernels.

The Mosaic family remains experimental and is intended as an opt-in preview while
GPU bring-up work is paused between hardware access windows.
"""

import json
import math
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from nanovllm_jax.engine.decode_schedule import get_decode_schedule_packet

# Check if Pallas with Mosaic GPU is available
try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu
    PALLAS_AVAILABLE = True
except ImportError:
    PALLAS_AVAILABLE = False
    pl = None
    plgpu = None

# Optional Mosaic GPU kernels
try:
    from . import mosaic_gpu_attention as mosaic_attn
    MOSAIC_AVAILABLE = mosaic_attn.MOSAIC_AVAILABLE
except ImportError:
    MOSAIC_AVAILABLE = False
    mosaic_attn = None

_MOSAIC_SHAPE_KEY = tuple[int, int, int, int]
_MOSAIC_VARIANT_KEY = tuple[int, int, int, int, int, int, str]
_MOSAIC_FAILURE_KEY = tuple[str, int, int, int, int, int, int, int]


@dataclass
class AttentionBackendRuntimeState:
    """Mutable backend dispatch state owned by the active runtime."""

    use_mosaic_paged_decode: bool
    use_blockwise_decode: bool
    prefill_disabled_reason: str | None = None
    probe_ok: bool = False
    probe_attempted: bool = False
    baseline_disabled_reason: str | None = None
    latency_disabled_reason: str | None = None
    throughput_disabled_reason: str | None = None
    throughput_v2_disabled_reason: str | None = None
    variant_selection_cache: dict[_MOSAIC_VARIANT_KEY, str] = field(default_factory=dict)
    failure_cache: dict[_MOSAIC_FAILURE_KEY, str] = field(default_factory=dict)
    tile_cache: dict[tuple[int, int, int, int, int], object] = field(default_factory=dict)

    def reset_decode_transients(self) -> None:
        self.probe_ok = False
        self.probe_attempted = False
        self.baseline_disabled_reason = None
        self.latency_disabled_reason = None
        self.throughput_disabled_reason = None
        self.throughput_v2_disabled_reason = None
        self.variant_selection_cache.clear()
        self.failure_cache.clear()
        self.tile_cache.clear()

    def reset_all_transients(self) -> None:
        self.prefill_disabled_reason = None
        self.reset_decode_transients()


def create_attention_backend_runtime_state() -> AttentionBackendRuntimeState:
    return AttentionBackendRuntimeState(
        use_mosaic_paged_decode=(
            os.environ.get("NANOVLLM_JAX_USE_MOSAIC_DECODE", "0") == "1"
        ),
        use_blockwise_decode=(
            os.environ.get("NANOVLLM_JAX_USE_BLOCKWISE_DECODE", "1") == "1"
        ),
    )


_ACTIVE_BACKEND_RUNTIME_STATE = create_attention_backend_runtime_state()


def get_attention_backend_runtime_state() -> AttentionBackendRuntimeState:
    return _ACTIVE_BACKEND_RUNTIME_STATE


def set_attention_backend_runtime_state(
    runtime_state: AttentionBackendRuntimeState | None,
) -> AttentionBackendRuntimeState:
    global _ACTIVE_BACKEND_RUNTIME_STATE
    _ACTIVE_BACKEND_RUNTIME_STATE = (
        runtime_state if runtime_state is not None else create_attention_backend_runtime_state()
    )
    return _ACTIVE_BACKEND_RUNTIME_STATE


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

# Mosaic kernel tuning defaults (overridable via configure_attention_backends)
_MOSAIC_BLOCK_Q = 64
# H100 quick screens on 2026-04-03 showed the current Mosaic families were
# materially slower at block_kv=256 than at block_kv=64 on the tested shapes.
_MOSAIC_BLOCK_KV = 64
_MOSAIC_MAX_CONCURRENT_STEPS = 2
_MOSAIC_MIN_DECODE_BATCH = _env_int(
    "NANOVLLM_JAX_INTERNAL_MOSAIC_MIN_DECODE_BATCH",
    512,
)  # min padded batch for auto-selection
_MOSAIC_THROUGHPUT_SPLIT_K = 0
_MOSAIC_THROUGHPUT_NUM_STAGES = 2
_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH = _env_int(
    "NANOVLLM_JAX_INTERNAL_MOSAIC_THROUGHPUT_MIN_DECODE_BATCH",
    128,
)
_MOSAIC_THROUGHPUT_RESCALE_THRESHOLD = 1.0
_MOSAIC_THROUGHPUT_AUTOTUNE = False
_MOSAIC_DECODE_KERNEL_FAMILY = os.environ.get(
    "NANOVLLM_JAX_MOSAIC_DECODE_KERNEL", "auto"
).strip().lower()
_MOSAIC_DECODE_FAMILY_TABLE_PATH = (
    os.environ.get("NANOVLLM_JAX_MOSAIC_DECODE_KERNEL_TABLE_PATH", "").strip() or None
)
_MOSAIC_DECODE_FAMILY_TABLE: dict[tuple[object, ...], str] | None = None
_MOSAIC_VARIANT_SELECTION_CACHE_MAX = 256
_MOSAIC_DECODE_FAILURE_CACHE_MAX = 256
_THROUGHPUT_V2_CANARY_SHAPE_TABLE: dict[_MOSAIC_VARIANT_KEY, str] = {
    (512, 128, 16, 256, 16, 8, "bfloat16"): "throughput_v2",
    (512, 128, 24, 256, 16, 8, "bfloat16"): "throughput_v2",
    (512, 128, 32, 256, 16, 8, "bfloat16"): "throughput_v2",
    (512, 128, 48, 256, 16, 8, "bfloat16"): "throughput_v2",
    (512, 128, 64, 256, 16, 8, "bfloat16"): "throughput_v2",
    (1024, 128, 16, 256, 16, 8, "bfloat16"): "throughput_v2",
    (1024, 128, 32, 256, 16, 8, "bfloat16"): "throughput_v2",
    (2048, 128, 16, 256, 16, 8, "bfloat16"): "throughput_v2",
    (2048, 128, 32, 256, 16, 8, "bfloat16"): "throughput_v2",
    (4096, 128, 16, 256, 16, 8, "bfloat16"): "throughput_v2",
}


def _normalize_mosaic_variant(raw: object) -> str:
    variant = str(raw).strip().lower()
    return (
        variant
        if variant in {"auto", "baseline", "latency", "throughput", "throughput_v2"}
        else "auto"
    )


def _dtype_key(dtype: object) -> str:
    return str(dtype).replace("'", "").strip().lower()


def _parse_shape_key(raw_key: str) -> tuple[object, ...] | None:
    fields: dict[str, str] = {}
    try:
        for token in raw_key.split(","):
            if "=" not in token:
                return None
            key, value = token.split("=", 1)
            fields[key.strip()] = value.strip()
        required = (
            int(fields["batch"]),
            int(fields["head_dim"]),
            int(fields["blocks"]),
            int(fields["block_size"]),
        )
        if {"num_heads", "num_kv_heads", "dtype"} <= fields.keys():
            return (
                *required,
                int(fields["num_heads"]),
                int(fields["num_kv_heads"]),
                _dtype_key(fields["dtype"]),
            )
        return required
    except Exception:
        return None


def _load_mosaic_variant_table_if_needed() -> None:
    global _MOSAIC_DECODE_FAMILY_TABLE
    if _MOSAIC_DECODE_FAMILY_TABLE is not None:
        return
    parsed: dict[tuple[int, int, int, int], str] = {}
    path = _MOSAIC_DECODE_FAMILY_TABLE_PATH
    if path is not None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for raw_key, raw_value in payload.items():
                    if not isinstance(raw_key, str):
                        continue
                    shape_key = _parse_shape_key(raw_key)
                    if shape_key is None:
                        continue
                    variant = _normalize_mosaic_variant(raw_value)
                    if variant != "auto":
                        parsed[shape_key] = variant
        except Exception:
            parsed = {}
    _MOSAIC_DECODE_FAMILY_TABLE = parsed


def _lookup_mosaic_variant_table(
    shape_key: _MOSAIC_SHAPE_KEY,
    strict_shape_key: _MOSAIC_VARIANT_KEY,
) -> str | None:
    _load_mosaic_variant_table_if_needed()
    if _MOSAIC_DECODE_FAMILY_TABLE is not None:
        override = _MOSAIC_DECODE_FAMILY_TABLE.get(strict_shape_key)
        if override is not None:
            return override
        override = _MOSAIC_DECODE_FAMILY_TABLE.get(shape_key)
        if override is not None:
            return override
    if (
        mosaic_attn is not None
        and getattr(
            mosaic_attn,
            "_should_use_throughput_v2_mosaic_kernel",
            lambda: False,
        )()
    ):
        return _THROUGHPUT_V2_CANARY_SHAPE_TABLE.get(strict_shape_key)
    return None


def _heuristic_mosaic_variant(
    *,
    padded_batch: int,
    head_dim: int,
    max_blocks_per_seq: int,
) -> str:
    if max_blocks_per_seq >= 32 and padded_batch >= 512:
        return "throughput"
    if max_blocks_per_seq == 24 and padded_batch >= 256:
        return "baseline"
    if max_blocks_per_seq >= 16:
        return "throughput"
    return "latency"


def _select_mosaic_decode_variant(
    *,
    requested_variant: str,
    padded_batch: int,
    head_dim: int,
    max_blocks_per_seq: int,
    block_size: int,
    num_heads: int = 0,
    num_kv_heads: int = 0,
    dtype: object = "unknown",
) -> str:
    state = get_attention_backend_runtime_state()
    requested = _normalize_mosaic_variant(requested_variant)
    if requested != "auto":
        return requested

    shape_key = (padded_batch, head_dim, max_blocks_per_seq, block_size)
    strict_shape_key = (
        padded_batch,
        head_dim,
        max_blocks_per_seq,
        block_size,
        int(num_heads),
        int(num_kv_heads),
        _dtype_key(dtype),
    )
    cached = state.variant_selection_cache.get(strict_shape_key)
    if cached is not None:
        return cached

    variant = _lookup_mosaic_variant_table(shape_key, strict_shape_key)
    if variant is None:
        variant = _heuristic_mosaic_variant(
            padded_batch=padded_batch,
            head_dim=head_dim,
            max_blocks_per_seq=max_blocks_per_seq,
        )
    state.variant_selection_cache[strict_shape_key] = variant
    if len(state.variant_selection_cache) > _MOSAIC_VARIANT_SELECTION_CACHE_MAX:
        state.variant_selection_cache.clear()
    return variant


def _make_mosaic_failure_key(
    *,
    variant: str,
    batch_size: int,
    padded_batch: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_blocks_per_seq: int,
    block_size: int,
) -> tuple[str, int, int, int, int, int, int, int]:
    return (
        variant,
        batch_size,
        padded_batch,
        num_heads,
        num_kv_heads,
        head_dim,
        max_blocks_per_seq,
        block_size,
    )


def _record_mosaic_failure(
    key: _MOSAIC_FAILURE_KEY,
    reason: str,
) -> None:
    state = get_attention_backend_runtime_state()
    state.failure_cache[key] = reason
    if len(state.failure_cache) > _MOSAIC_DECODE_FAILURE_CACHE_MAX:
        state.failure_cache.clear()


def configure_attention_backends(
    config,
    runtime_state: AttentionBackendRuntimeState | None = None,
) -> None:
    """Apply Config-driven attention backend selection.

    Call once at engine startup (from ModelRunner.__init__).  When the
    config field is ``"auto"`` the env-var / module default is kept.
    Internal Mosaic tuning stays private to this module and profiling tools.
    """
    state = (
        set_attention_backend_runtime_state(runtime_state)
        if runtime_state is not None
        else get_attention_backend_runtime_state()
    )

    backend = str(getattr(config, "decode_attention_backend", "auto")).strip().lower()
    valid_backends = {"auto", "mosaic", "blockwise"}
    if backend not in valid_backends:
        raise ValueError(
            "decode_attention_backend must be one of: auto|mosaic|blockwise"
        )
    if backend != "auto":
        state.use_mosaic_paged_decode = (backend == "mosaic")
        state.use_blockwise_decode = (backend == "blockwise")

    state.reset_all_transients()


def _maybe_run_mosaic_prefill(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    max_seqlen: int | jax.Array,
    scale: float,
):
    """Attempt Mosaic prefill kernel, falling back silently on failure."""
    state = get_attention_backend_runtime_state()

    if not MOSAIC_AVAILABLE or mosaic_attn is None:
        return None

    if state.prefill_disabled_reason is not None:
        return None

    try:
        max_len_int = int(max_seqlen)
    except (TypeError, ValueError):
        # max_seqlen may be a tracer when called under jit; skip Mosaic in that case.
        return None

    block_q = 64
    block_kv = 64

    # WGMMA requires M >= 64; skip Mosaic if max sequence length is too small.
    if max_len_int < block_q:
        return None

    if block_kv > k.shape[0]:
        block_kv = max(64, min(k.shape[0], block_kv))

    max_steps_hint = max(2, min(4, max(2, max_len_int // max(1, block_kv))))

    try:
        mosaic_config = mosaic_attn.MosaicAttentionConfig(
            block_q=block_q,
            block_kv=block_kv,
            max_concurrent_steps=max_steps_hint,
            use_schedule_barrier=True,
            num_compute_wgs=2,
        )
        mosaic_out, _ = mosaic_attn.prefill_attention_mosaic_api(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_len_int,
            scale=scale,
            config=mosaic_config,
        )
        return mosaic_out
    except (RuntimeError, ValueError) as exc:
        state.prefill_disabled_reason = str(exc)
        return None


def _maybe_run_mosaic_decode(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    block_size: int | jax.Array,
    decode_schedule_token: int = 0,
):
    """Attempt Mosaic decode kernel, falling back silently on failure.

    Includes auto-selection: Mosaic's WGMMA pipeline has fixed overhead that
    only pays off at large batch sizes.  Kernel benchmarks (H100, 1-block):
      hd=64:  Mosaic wins at padded_batch >= 512 (1.2x), slower below
      hd=128: Mosaic wins at padded_batch >= 512 (1.7x), slower below
    The threshold is controlled by module-private tuning used by the
    profiling scripts and environment overrides.

    Tile metadata is cached on the runner-owned decode schedule packet during
    tracing to avoid redundant graph construction across attention layers.
    """
    state = get_attention_backend_runtime_state()
    schedule_packet = get_decode_schedule_packet(decode_schedule_token)

    if not MOSAIC_AVAILABLE or mosaic_attn is None:
        return None

    try:
        block_size_int = int(block_size)
    except (TypeError, ValueError):
        # block_size may be a tracer when called under jit; skip Mosaic in that case.
        return None

    # Use module-private tuning params (overridden by profiling tools/env vars).
    block_q = _MOSAIC_BLOCK_Q
    block_kv = _MOSAIC_BLOCK_KV if _MOSAIC_BLOCK_KV % 64 == 0 else 64

    # --- Auto-selection: skip Mosaic when kernel is expected to be slower ---
    min_padded = _MOSAIC_MIN_DECODE_BATCH
    padded_batch = ((q.shape[0] + block_q - 1) // block_q) * block_q
    if min_padded > 0:
        if padded_batch < min_padded:
            return None

    # Prefer full-page KV tiles; fall back to 64-token tiles if misaligned.
    if block_kv > block_size_int:
        block_kv = max(64, (block_size_int // 64) * 64)
    elif block_kv == 0:
        block_kv = block_size_int if block_size_int % 64 == 0 else 64

    if block_size_int % block_kv != 0:
        return None

    latency_shape_eligible = (q.shape[-1] >= 128)
    throughput_shape_eligible = (
        latency_shape_eligible
        and block_tables.shape[1] >= 16
        and padded_batch >= _MOSAIC_THROUGHPUT_MIN_DECODE_BATCH
    )

    requested_variant = _normalize_mosaic_variant(_MOSAIC_DECODE_KERNEL_FAMILY)
    selected_variant = _select_mosaic_decode_variant(
        requested_variant=requested_variant,
        padded_batch=padded_batch,
        head_dim=q.shape[-1],
        max_blocks_per_seq=block_tables.shape[1],
        block_size=block_size_int,
        num_heads=q.shape[1],
        num_kv_heads=k_cache.shape[2],
        dtype=q.dtype,
    )

    if selected_variant == "throughput_v2":
        variant_chain = ("throughput_v2", "throughput", "baseline")
    elif selected_variant == "throughput":
        if requested_variant == "auto":
            variant_chain = ("throughput", "latency", "baseline")
        else:
            variant_chain = ("throughput", "baseline")
    elif selected_variant == "latency":
        variant_chain = ("latency", "baseline")
    else:
        variant_chain = ("baseline",)

    for candidate in variant_chain:
        failure_key = _make_mosaic_failure_key(
            variant=candidate,
            batch_size=q.shape[0],
            padded_batch=padded_batch,
            num_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim=q.shape[-1],
            max_blocks_per_seq=block_tables.shape[1],
            block_size=block_size_int,
        )
        if failure_key in state.failure_cache:
            continue

        if candidate == "throughput_v2":
            if not throughput_shape_eligible:
                continue
            try:
                throughput_v2_block_kv = 64 if (block_size_int % 64 == 0) else block_kv
                throughput_v2_config = mosaic_attn.MosaicAttentionConfig(
                    block_q=64,
                    block_kv=throughput_v2_block_kv,
                    max_concurrent_steps=2,
                    use_schedule_barrier=False,
                    num_compute_wgs=1,
                )
                out = mosaic_attn.paged_decode_attention_mosaic_throughput_v2(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    scale=scale,
                    block_size=block_size_int,
                    config=throughput_v2_config,
                    split_k=_MOSAIC_THROUGHPUT_SPLIT_K,
                    prepared_metadata_cache=(
                        schedule_packet.metadata_cache_for_family("throughput_v2")
                        if schedule_packet is not None
                        else None
                    ),
                )
                return out
            except Exception as exc:
                state.throughput_v2_disabled_reason = str(exc)
                _record_mosaic_failure(failure_key, state.throughput_v2_disabled_reason)
                continue

        if candidate == "throughput":
            if not throughput_shape_eligible:
                continue
            try:
                throughput_config = mosaic_attn.MosaicAttentionConfig(
                    block_q=block_q,
                    block_kv=block_kv,
                    max_concurrent_steps=max(2, _MOSAIC_THROUGHPUT_NUM_STAGES),
                    use_schedule_barrier=True,
                    num_compute_wgs=2,
                )
                out = mosaic_attn.paged_decode_attention_mosaic_throughput(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    scale=scale,
                    block_size=block_size_int,
                    config=throughput_config,
                    split_k=_MOSAIC_THROUGHPUT_SPLIT_K,
                    rescale_threshold=_MOSAIC_THROUGHPUT_RESCALE_THRESHOLD,
                    autotune=_MOSAIC_THROUGHPUT_AUTOTUNE,
                    prepared_metadata_cache=(
                        schedule_packet.metadata_cache_for_family("throughput")
                        if schedule_packet is not None
                        else None
                    ),
                )
                return out
            except Exception as exc:
                state.throughput_disabled_reason = str(exc)
                _record_mosaic_failure(failure_key, state.throughput_disabled_reason)
                continue

        if candidate == "latency":
            if not latency_shape_eligible:
                continue
            try:
                latency_config = mosaic_attn.MosaicAttentionConfig(
                    block_q=block_q,
                    block_kv=block_kv,
                    max_concurrent_steps=_MOSAIC_MAX_CONCURRENT_STEPS,
                    use_schedule_barrier=True,
                    num_compute_wgs=2,
                )
                out = mosaic_attn.paged_decode_attention_mosaic_latency(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    scale=scale,
                    block_size=block_size_int,
                    config=latency_config,
                    prepared_metadata_cache=(
                        schedule_packet.metadata_cache_for_family("latency")
                        if schedule_packet is not None
                        else None
                    ),
                )
                return out
            except Exception as exc:
                state.latency_disabled_reason = str(exc)
                _record_mosaic_failure(failure_key, state.latency_disabled_reason)
                continue

        if candidate == "baseline":
            if not _ensure_mosaic_decode_probe_ready(batch_size=q.shape[0], block_q=block_q):
                continue
            try:
                mosaic_config = mosaic_attn.MosaicAttentionConfig(
                    block_q=block_q,
                    block_kv=block_kv,
                    max_concurrent_steps=_MOSAIC_MAX_CONCURRENT_STEPS,
                    use_schedule_barrier=True,
                    num_compute_wgs=2,
                )

                metadata_key = (
                    "baseline",
                    q.shape[0],
                    block_q,
                    block_kv,
                    block_size_int,
                )
                if schedule_packet is not None:
                    metadata = schedule_packet.get_or_create_metadata(
                        "baseline",
                        metadata_key,
                        lambda: mosaic_attn.prepare_decode_metadata(
                            block_tables,
                            context_lens,
                            q.shape[0],
                            block_q,
                            block_size_int,
                            block_kv,
                            include_unused_fields=False,
                        ),
                    )
                else:
                    metadata = mosaic_attn.prepare_decode_metadata(
                        block_tables,
                        context_lens,
                        q.shape[0],
                        block_q,
                        block_size_int,
                        block_kv,
                        include_unused_fields=False,
                    )

                out = mosaic_attn.batched_decode_attention_mosaic(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    scale=scale,
                    config=mosaic_config,
                    metadata=metadata,
                )
                return out
            except Exception as exc:
                state.baseline_disabled_reason = str(exc)
                _record_mosaic_failure(failure_key, state.baseline_disabled_reason)
                continue

    return None


def _probe_mosaic_decode_startup() -> bool:
    """Compile+run a representative Mosaic decode once at startup."""
    state = get_attention_backend_runtime_state()

    if not state.use_mosaic_paged_decode:
        return False

    if not MOSAIC_AVAILABLE or mosaic_attn is None:
        state.baseline_disabled_reason = "Mosaic backend is unavailable."
        return False

    if jax.default_backend() != "gpu":
        state.baseline_disabled_reason = (
            "Mosaic decode requires GPU backend; current backend is "
            f"{jax.default_backend()}."
        )
        return False

    # Probe shape mirrors production decode constraints while keeping compile cost
    # bounded. If this fails, we keep the default blockwise path.
    probe_batch = max(64, _MOSAIC_BLOCK_Q)
    probe_num_heads = 8
    probe_num_kv_heads = 8
    probe_head_dim = 64
    probe_block_size = 256
    probe_max_blocks = 2
    probe_num_blocks = probe_batch * probe_max_blocks

    q = jnp.zeros((probe_batch, probe_num_heads, probe_head_dim), dtype=jnp.float16)
    k_cache = jnp.zeros(
        (probe_num_blocks, probe_block_size, probe_num_kv_heads, probe_head_dim),
        dtype=jnp.float16,
    )
    v_cache = jnp.zeros_like(k_cache)
    block_tables = jnp.arange(
        probe_batch * probe_max_blocks, dtype=jnp.int32
    ).reshape(probe_batch, probe_max_blocks)
    context_lens = jnp.full((probe_batch,), probe_block_size, dtype=jnp.int32)
    scale = float(1.0 / math.sqrt(probe_head_dim))

    try:
        # Call the kernel directly — bypass auto-selection heuristic which
        # would reject this small probe batch.  The probe tests whether the
        # kernel *works*, not whether it's *fast*.
        probe_config = mosaic_attn.MosaicAttentionConfig(
            block_q=_MOSAIC_BLOCK_Q,
            block_kv=probe_block_size if probe_block_size % 64 == 0 else 64,
            max_concurrent_steps=_MOSAIC_MAX_CONCURRENT_STEPS,
            use_schedule_barrier=True,
            num_compute_wgs=2,
        )
        out = mosaic_attn.batched_decode_attention_mosaic(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            config=probe_config,
        )
        jax.block_until_ready(out)
        return True
    except Exception as exc:
        state.baseline_disabled_reason = f"Startup probe failed: {exc}"
        return False


def _ensure_mosaic_decode_probe_ready(batch_size: int, block_q: int) -> bool:
    """Ensure Mosaic decode probe is run when needed, and only when needed."""
    state = get_attention_backend_runtime_state()

    if state.probe_ok:
        return True

    # Previous probe attempt failed; don't retry in the same process.
    if state.probe_attempted:
        return False

    # In auto-selection mode, avoid probing if this shape can never use Mosaic.
    min_padded = _MOSAIC_MIN_DECODE_BATCH
    if min_padded > 0:
        padded_batch = ((batch_size + block_q - 1) // block_q) * block_q
        if padded_batch < min_padded:
            return False

    state.probe_attempted = True
    state.probe_ok = _probe_mosaic_decode_startup()
    return state.probe_ok


class PagedAttentionConfig(NamedTuple):
    """Configuration for paged attention kernel.
    
    Attributes:
        block_size: Tokens per KV-cache block (must match BlockManager).
        block_kv: KV tile size for pipelining (typically 64 or 128).
        max_concurrent_steps: Pipeline depth (typically 2-4).
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Dimension per attention head.
    """
    block_size: int = 256  # Must match BlockManager block_size
    block_kv: int = 64     # KV tile size for kernel (must divide block_size)
    max_concurrent_steps: int = 2
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 64


def _check_pallas_available():
    """Check if Pallas with Mosaic GPU backend is available."""
    if not PALLAS_AVAILABLE:
        raise RuntimeError(
            "Pallas with Mosaic GPU backend is not available. "
            "Make sure you have JAX installed with GPU support and the correct version."
        )


# =============================================================================
# Paged Decode Attention - Vectorized Implementation
# =============================================================================

@partial(jax.jit, static_argnums=(5, 6))
def paged_decode_attention_vectorized(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    block_size: int,
) -> jax.Array:
    """Optimized paged decode attention using vectorized operations.
    
    This version avoids Python loops and uses JAX's vectorization to achieve
    better GPU utilization. The key optimization is:
    1. Gather all relevant K/V blocks in one operation
    2. Compute attention in parallel across all blocks
    3. Use online softmax aggregation with vmap
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: Block indices for each sequence [batch_size, max_blocks_per_seq].
        context_lens: Context length for each sequence [batch_size].
        scale: Softmax scale factor.
        block_size: Tokens per KV-cache block.
    
    Returns:
        Output tensor [batch_size, num_heads, head_dim].
    """
    batch_size, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k_cache.shape
    max_blocks_per_seq = block_tables.shape[1]
    max_context_len = max_blocks_per_seq * block_size
    
    # GQA ratio
    q_heads_per_kv_head = num_heads // num_kv_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    
    # Clamp block indices to valid range
    safe_block_tables = jnp.clip(block_tables, min=0, max=k_cache.shape[0] - 1)
    
    # Gather K/V blocks: [batch, max_blocks] -> [batch, max_blocks, block_size, kv_heads, dim]
    k_gathered = k_cache[safe_block_tables]  # [batch, max_blocks, block_size, kv_heads, dim]
    v_gathered = v_cache[safe_block_tables]
    
    # Reshape to [batch, max_context_len, kv_heads, dim]
    k_flat = k_gathered.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    v_flat = v_gathered.reshape(batch_size, max_context_len, num_kv_heads, head_dim)

    # Expand KV heads for GQA: [batch, seq, kv_heads, dim] -> [batch, seq, num_heads, dim]
    # Each KV head is repeated q_heads_per_kv_head times.
    k_expanded = jnp.repeat(k_flat, q_heads_per_kv_head, axis=2)  # [batch, seq, num_heads, dim]
    v_expanded = jnp.repeat(v_flat, q_heads_per_kv_head, axis=2)

    # Cast to float32 for stable softmax math.
    q_f32 = q.astype(jnp.float32)  # [batch, num_heads, dim]
    k_f32 = k_expanded.astype(jnp.float32)  # [batch, seq, num_heads, dim]
    v_f32 = v_expanded.astype(jnp.float32)

    # Compute attention scores: Q @ K^T
    # q: [batch, heads, dim] -> [batch, heads, 1, dim]
    # k: [batch, seq, heads, dim] -> [batch, heads, seq, dim] (transpose)
    q_expanded = q_f32[:, :, None, :]  # [batch, heads, 1, dim]
    k_transposed = jnp.transpose(k_f32, (0, 2, 1, 3))  # [batch, heads, seq, dim]

    # Scaled dot-product: [batch, heads, 1, dim] @ [batch, heads, dim, seq] = [batch, heads, 1, seq]
    scores = jnp.matmul(q_expanded, jnp.transpose(k_transposed, (0, 1, 3, 2))) * scale
    scores = scores.squeeze(2)  # [batch, heads, seq]
    
    # Create attention mask based on context_lens
    positions = jnp.arange(max_context_len)[None, None, :]  # [1, 1, seq]
    mask = positions < context_lens[:, None, None]  # [batch, 1, seq]
    
    # Apply mask (use large negative value for masked positions)
    scores = jnp.where(mask, scores, jnp.float32(-1e9))
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = jnp.exp(scores - scores_max)
    scores_exp = jnp.where(mask, scores_exp, 0.0)
    scores_sum = scores_exp.sum(axis=-1, keepdims=True)
    attn_weights = scores_exp / (scores_sum + 1e-9)  # [batch, heads, seq]

    # Weighted sum of values
    # attn_weights: [batch, heads, seq] -> [batch, heads, seq, 1]
    # v: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    v_transposed = jnp.transpose(v_f32, (0, 2, 1, 3))  # [batch, heads, seq, dim]
    attn_weights_expanded = attn_weights[:, :, :, None]  # [batch, heads, seq, 1]

    # Element-wise multiply and sum over seq dimension
    output = (attn_weights_expanded * v_transposed).sum(axis=2)  # [batch, heads, dim]

    return output.astype(q.dtype)


@partial(jax.jit, static_argnums=(5, 6))
def paged_decode_attention_blockwise(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    block_size: int,
) -> jax.Array:
    """Paged decode attention with streaming online softmax (no dense gather).

    Key idea: iterate over KV blocks (static `max_blocks_per_seq`) and update
    FlashAttention-style online softmax state. This avoids materializing the full
    `[batch, max_context_len, ...]` K/V tensors and avoids expanding KV heads for GQA.
    """
    batch_size, num_heads, head_dim = q.shape
    num_blocks, cache_block_size, num_kv_heads, head_dim_k = k_cache.shape
    if cache_block_size != block_size:
        raise ValueError(f"block_size mismatch: cache={cache_block_size} arg={block_size}")
    if head_dim_k != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim} k={head_dim_k}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

    q_heads_per_kv_head = num_heads // num_kv_heads
    qg = q.reshape(batch_size, num_kv_heads, q_heads_per_kv_head, head_dim).astype(jnp.float32)

    # Online softmax state (natural log space).
    m = jnp.full((batch_size, num_kv_heads, q_heads_per_kv_head), -jnp.inf, dtype=jnp.float32)
    l = jnp.zeros((batch_size, num_kv_heads, q_heads_per_kv_head), dtype=jnp.float32)
    acc = jnp.zeros((batch_size, num_kv_heads, q_heads_per_kv_head, head_dim), dtype=jnp.float32)

    log2e = jnp.float32(math.log2(math.e))
    max_blocks_per_seq = block_tables.shape[1]

    def body(block_idx, carry):
        acc, m, l = carry

        # [batch]
        phys = jnp.clip(block_tables[:, block_idx], min=0, max=num_blocks - 1)

        # Gather one KV block: [batch, block_size, kv_heads, head_dim]
        kb = k_cache[phys]
        vb = v_cache[phys]
        kb = jnp.transpose(kb, (0, 2, 1, 3))  # [batch, kv_heads, block_size, head_dim]
        vb = jnp.transpose(vb, (0, 2, 1, 3))

        # QK^T for this block: [batch, kv_heads, q_per_kv, block_size]
        logits = jnp.einsum(
            "bgqd,bgsd->bgqs",
            qg,
            kb,
            preferred_element_type=jnp.float32,
        ) * scale

        # Mask within-block tokens beyond context length.
        start = jnp.int32(block_idx * block_size)
        valid = jnp.clip(context_lens - start, min=0, max=block_size).astype(jnp.int32)
        pos = jnp.arange(block_size, dtype=jnp.int32)[None, None, None, :]
        mask = pos < valid[:, None, None, None]
        logits = jnp.where(mask, logits, -jnp.inf)

        # Online softmax update (use exp2 for throughput).
        m_curr = logits.max(axis=-1)
        m_next = jnp.maximum(m, m_curr)
        corr = jnp.exp2((m - m_next) * log2e)
        l_corr = l * corr
        p = jnp.exp2((logits - m_next[..., None]) * log2e)
        p = jnp.where(mask, p, 0.0)
        l_next = l_corr + p.sum(axis=-1)
        acc = acc * corr[..., None] + jnp.einsum(
            "bgqs,bgsd->bgqd",
            p,
            vb,
            preferred_element_type=jnp.float32,
        )
        return (acc, m_next, l_next)

    acc, m, l = lax.fori_loop(0, max_blocks_per_seq, body, (acc, m, l))
    out = acc / jnp.maximum(l[..., None], 1e-9)
    out = out.reshape(batch_size, num_heads, head_dim).astype(q.dtype)
    return out


# =============================================================================
# Paged Decode Attention Kernel (Pallas with loop - slower, for reference)
# =============================================================================

def paged_decode_attention_kernel(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    config: PagedAttentionConfig,
) -> jax.Array:
    """Paged decode attention using Pallas kernel.
    
    Computes attention directly over paged KV-cache without gathering into dense tensors.
    Uses FlashAttention-style online softmax for numerical stability and memory efficiency.
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: Block indices for each sequence [batch_size, max_blocks_per_seq].
        context_lens: Context length for each sequence [batch_size].
        scale: Softmax scale factor (typically 1/sqrt(head_dim)).
        config: Kernel configuration.
    
    Returns:
        Output tensor [batch_size, num_heads, head_dim].
    """
    _check_pallas_available()
    
    batch_size, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks_per_seq = block_tables.shape[1]
    
    # Validate configuration
    if head_dim != config.head_dim:
        raise ValueError(f"head_dim mismatch: {head_dim} vs {config.head_dim}")
    if num_heads != config.num_heads:
        raise ValueError(f"num_heads mismatch: {num_heads} vs {config.num_heads}")
    if num_kv_heads != config.num_kv_heads:
        raise ValueError(f"num_kv_heads mismatch: {num_kv_heads} vs {config.num_kv_heads}")
    if block_size != config.block_size:
        raise ValueError(f"block_size mismatch: {block_size} vs {config.block_size}")
    
    # GQA: number of query heads per KV head
    q_heads_per_kv_head = num_heads // num_kv_heads
    
    # For simplicity, we use a straightforward grid: (batch, heads)
    # Each program computes attention for one (batch, head) pair
    grid = (batch_size, num_heads)
    
    def kernel_fn(
        q_ref,           # [batch, heads, dim]
        k_cache_ref,     # [num_blocks, block_size, kv_heads, dim]
        v_cache_ref,     # [num_blocks, block_size, kv_heads, dim]
        block_tables_ref,  # [batch, max_blocks]
        context_lens_ref,  # [batch]
        out_ref,         # [batch, heads, dim]
    ):
        batch_idx = pl.program_id(0)
        head_idx = pl.program_id(1)
        
        # Map query head to KV head (for GQA)
        kv_head_idx = head_idx // q_heads_per_kv_head
        
        # Load query for this (batch, head) and cast to float32
        q_vec = q_ref[batch_idx, head_idx, :].astype(jnp.float32)  # [head_dim]
        
        # Get context length for this sequence
        context_len = context_lens_ref[batch_idx]
        
        # Number of blocks to process
        num_context_blocks = (context_len + block_size - 1) // block_size
        
        # Initialize online softmax state (FlashAttention algorithm)
        # m_i: running max of attention logits (in log2 space for FMA)
        # l_i: running sum of exp(logits - m_i)
        # acc: running weighted sum of values
        m_i = jnp.float32(-1e9)  # Start with very negative max
        l_i = jnp.float32(0.0)
        acc = jnp.zeros((head_dim,), dtype=jnp.float32)
        
        # Scale factor in log2 space for efficient exp2 computation
        log2e = jnp.float32(math.log2(math.e))
        scale_log2e = scale * log2e
        
        # Reshape query for matrix multiply: [head_dim] -> [1, head_dim]
        q_vec_2d = q_vec[None, :]  # [1, head_dim]
        
        # Process each block in the sequence
        def process_block(block_idx, carry):
            m_i, l_i, acc = carry
            
            # Get physical block index from block table
            physical_block = block_tables_ref[batch_idx, block_idx]
            
            # Calculate valid tokens in this block
            block_start = block_idx * block_size
            block_end = jnp.minimum(block_start + block_size, context_len)
            valid_tokens = block_end - block_start
            
            # Load K and V for this block and cast to float32 for computation
            # k_block: [block_size, head_dim]
            # v_block: [block_size, head_dim]
            k_block = k_cache_ref[physical_block, :, kv_head_idx, :].astype(jnp.float32)
            v_block = v_cache_ref[physical_block, :, kv_head_idx, :].astype(jnp.float32)
            
            # Compute attention scores: Q @ K^T with scaling
            # q_vec_2d: [1, head_dim], k_block: [block_size, head_dim]
            # Use matmul: Q @ K^T = [1, head_dim] @ [head_dim, block_size] = [1, block_size]
            scores = jnp.matmul(q_vec_2d, k_block.T) * scale_log2e  # [1, block_size]
            scores = scores.squeeze(0)  # [block_size]
            
            # Mask invalid positions (tokens beyond context_len within this block)
            token_indices = jnp.arange(block_size)
            valid_mask = token_indices < valid_tokens
            scores = jnp.where(valid_mask, scores, jnp.float32(-1e9))
            
            # Online softmax update (FlashAttention algorithm)
            # Find max of current block
            m_ij = scores.max()
            
            # New global max
            m_new = jnp.maximum(m_i, m_ij)
            
            # Rescaling factors for numerical stability
            alpha = jnp.exp2(m_i - m_new)  # Rescale previous accumulator
            beta = jnp.exp2(m_ij - m_new)  # Scale for current block
            
            # Compute softmax weights for this block
            p = jnp.exp2(scores - m_new)  # [block_size], unnormalized
            p = jnp.where(valid_mask, p, 0.0)  # Zero out invalid positions
            
            # Update running sum
            l_new = alpha * l_i + p.sum()
            
            # Update accumulator: acc = alpha * acc + p @ V
            # p: [block_size] -> p_2d: [1, block_size]
            # v_block: [block_size, head_dim]
            # Use matmul: [1, block_size] @ [block_size, head_dim] = [1, head_dim]
            p_2d = p[None, :]  # [1, block_size]
            pv = jnp.matmul(p_2d, v_block)  # [1, head_dim]
            acc_new = alpha * acc + pv.squeeze(0)  # [head_dim]
            
            return m_new, l_new, acc_new
        
        # Loop over blocks (use fori_loop for JIT compatibility)
        # Note: num_context_blocks is data-dependent, so we loop over max_blocks
        # and use conditionals to skip invalid blocks
        def cond_process_block(block_idx, carry):
            m_i, l_i, acc = carry
            
            # Only process if block is valid
            is_valid = block_idx < num_context_blocks
            
            # Process block or keep carry unchanged
            def do_process():
                return process_block(block_idx, (m_i, l_i, acc))
            
            def skip():
                return (m_i, l_i, acc)
            
            return lax.cond(is_valid, do_process, skip)
        
        # Process all potential blocks
        m_final, l_final, acc_final = lax.fori_loop(
            0, max_blocks_per_seq, cond_process_block, (m_i, l_i, acc)
        )
        
        # Normalize output
        out = acc_final / l_final
        
        # Write output
        out_ref[batch_idx, head_idx, :] = out.astype(out_ref.dtype)
    
    # Call kernel
    out_shape = jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), q.dtype)
    
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid=grid,
        interpret=False,  # Use GPU execution
    )(q, k_cache, v_cache, block_tables, context_lens)


# =============================================================================
# Optimized Paged Decode Attention (using Mosaic GPU features)
# =============================================================================

def paged_decode_attention_mosaic(
    q: jax.Array,           # [batch_size, num_heads, head_dim]
    k_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jax.Array,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: jax.Array,  # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array,  # [batch_size]
    scale: float,
    config: PagedAttentionConfig,
) -> jax.Array:
    """Optimized paged decode attention using Mosaic GPU features.
    
    This version uses:
    - SMEM for KV block caching
    - TMA for async memory transfers
    - WGMMA for TensorCore matmul (when dimensions align)
    
    Falls back to basic kernel if Mosaic GPU features are not available.
    """
    _check_pallas_available()
    
    # For now, use the basic kernel since Mosaic GPU requires careful
    # dimension alignment (M%64, N%8, K%swizzle) which is complex
    # for the variable-length paged attention case.
    # 
    # The basic Pallas kernel already provides significant speedup by
    # avoiding the gather-then-attend pattern.
    
    return paged_decode_attention_kernel(
        q, k_cache, v_cache, block_tables, context_lens, scale, config
    )


# =============================================================================
# High-Level API
# =============================================================================

@partial(jax.jit, static_argnums=(5, 6, 7))
def paged_attention(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    block_size: int,
    decode_schedule_token: int = 0,
) -> jax.Array:
    """JIT-compiled paged attention for decode phase.
    
    This is the main entry point for paged attention. It automatically
    selects the best implementation based on available hardware features.
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: Block indices for each sequence [batch_size, max_blocks_per_seq].
        context_lens: Context length for each sequence [batch_size].
        scale: Softmax scale factor.
        block_size: Tokens per KV-cache block.
    
    Returns:
        Output tensor [batch_size, num_heads, head_dim].
    """
    batch_size, num_heads, head_dim = q.shape
    _, block_size_cache, num_kv_heads, _ = k_cache.shape

    # Experimental Mosaic decode path:
    # - Must be explicitly enabled with NANOVLLM_JAX_USE_MOSAIC_DECODE=1.
    # - Uses a one-time startup compile/run probe, triggered lazily only when
    #   decode batch shapes are eligible for Mosaic auto-selection.
    # - Falls back to non-Mosaic implementations if runtime shapes/config are
    #   unsupported.
    state = get_attention_backend_runtime_state()

    if state.use_mosaic_paged_decode:
        mosaic_out = _maybe_run_mosaic_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            block_size=block_size,
            decode_schedule_token=decode_schedule_token,
        )
        if mosaic_out is not None:
            return mosaic_out

    if state.use_blockwise_decode:
        out = paged_decode_attention_blockwise(
            q, k_cache, v_cache, block_tables, context_lens, scale, block_size,
        )
        return out

    out = paged_decode_attention_vectorized(
        q, k_cache, v_cache, block_tables, context_lens, scale, block_size,
    )
    return out


def _paged_attention_fallback(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    scale: float,
    block_size: int,
) -> jax.Array:
    """Fallback implementation using standard JAX operations.
    
    This is slower but works on all JAX backends.
    """
    batch_size = block_tables.shape[0]
    max_blocks = block_tables.shape[1]
    max_context_len = max_blocks * block_size
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    
    # Gather K/V from cache (the slow path we're trying to avoid)
    safe_block_tables = jnp.clip(block_tables, min=0, max=k_cache.shape[0] - 1)
    gathered_k_blocks = k_cache[safe_block_tables]
    gathered_v_blocks = v_cache[safe_block_tables]
    
    gathered_k = gathered_k_blocks.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    gathered_v = gathered_v_blocks.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    
    # Create attention mask
    positions = jnp.arange(max_context_len)[None, :]
    mask = positions < context_lens[:, None]
    mask = mask[:, None, None, :]  # [batch, 1, 1, max_len]
    
    # Attention
    q = q[:, None, :, :]  # [batch, 1, heads, dim]
    output = jax.nn.dot_product_attention(
        q, gathered_k, gathered_v,
        mask=mask,
        scale=scale,
        implementation="cudnn" if (jax.default_backend() == "gpu" and q.dtype == jnp.float16) else None,
    )
    
    return output.squeeze(1)  # [batch, heads, dim]


# =============================================================================
# Paged Prefill Attention (for completeness)
# =============================================================================

def paged_prefill_attention(
    q: jax.Array,           # [total_tokens, num_heads, head_dim]
    k: jax.Array,           # [total_tokens, num_kv_heads, head_dim]
    v: jax.Array,           # [total_tokens, num_kv_heads, head_dim]
    cu_seqlens: jax.Array,  # [batch_size + 1]
    max_seqlen: int,
    scale: float,
) -> jax.Array:
    """Prefill attention with variable-length sequences.
    
    For prefill, we use standard Flash Attention since:
    1. K/V are freshly computed (not from cache)
    2. Variable sequence lengths are the main complexity
    
    This implementation uses padding + masking which is reasonably efficient.
    A more advanced implementation could use Pallas with ragged tensor support.
    
    Args:
        q: Query tensor [total_tokens, num_heads, head_dim].
        k: Key tensor [total_tokens, num_kv_heads, head_dim].
        v: Value tensor [total_tokens, num_kv_heads, head_dim].
        cu_seqlens: Cumulative sequence lengths [batch_size + 1].
        max_seqlen: Maximum sequence length.
        scale: Softmax scale factor.
    
    Returns:
        Output tensor [total_tokens, num_heads, head_dim].
    """
    # Attempt Mosaic path first; fall back silently on failure.
    mosaic_out = _maybe_run_mosaic_prefill(q, k, v, cu_seqlens, max_seqlen, scale)
    if mosaic_out is not None:
        return mosaic_out
    # Import the existing implementation
    from nanovllm_jax.layers.attention import variable_length_attention_prefill
    
    batch_size = cu_seqlens.shape[0] - 1
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    
    return variable_length_attention_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
        batch_size=batch_size,
    )
