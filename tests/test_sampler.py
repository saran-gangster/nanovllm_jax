"""Sampler behavior tests."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.sampler import Sampler


def test_sampler_uses_deterministic_argmax_for_near_greedy_temperatures() -> None:
    sampler = Sampler(rngs=nnx.Rngs(0))
    logits = jnp.asarray(
        [
            [0.1, 4.2, 0.3, -1.0],
            [1.5, 0.4, 2.1, 0.2],
        ],
        dtype=jnp.float32,
    )
    temperatures = jnp.asarray([1e-8, 5e-7], dtype=jnp.float32)

    out1 = np.asarray(sampler(logits, temperatures))
    out2 = np.asarray(sampler(logits, temperatures))
    expected = np.asarray(jnp.argmax(logits, axis=-1))

    assert np.array_equal(out1, expected)
    assert np.array_equal(out2, expected)


def test_sampler_remains_stochastic_when_any_temperature_is_sampling() -> None:
    sampler = Sampler(rngs=nnx.Rngs(0))
    logits = jnp.zeros((2, 8), dtype=jnp.float32)
    temperatures = jnp.asarray([1e-8, 0.7], dtype=jnp.float32)

    samples = [np.asarray(sampler(logits, temperatures)) for _ in range(8)]
    second_row_tokens = {int(sample[1]) for sample in samples}

    assert len(second_row_tokens) > 1, "sampling path should generate non-constant draws"
