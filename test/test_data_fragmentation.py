from random import uniform
import jax
import pytest
from jax import numpy as jnp

import data_fragmentation
import utils

settings = utils.hash_dict(
    {
        "data_dir": "data",
        "window_size": 2048,
        "hop_size": 256,
        "n_fft": 2048,
        "window_fn": "hamming_window",
        "n_mels": 256,
        "min_overlap": 0.5,
        "fragment_size": 5,
        "padding_mode": "edge",
        "begin_time_fn": "fixed",
    }
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def rngs(rng):
    return jax.random.split(rng, 2)


def assert_valid_begin_times(frag_intervals, begin_times):
    assert begin_times.shape == (len(frag_intervals),)
    assert (begin_times >= frag_intervals[:, 0]).all()
    assert (begin_times + settings["fragment_size"] <= frag_intervals[:, 1]).all()


uniform_begin_time_fn = data_fragmentation.get_uniform_begin_time_fn(settings)


@pytest.mark.parametrize(
    "frag_intervals", [jnp.array([[0, 5], [0, 7], [2, 8], [4, 10]])]
)
def test_uniform_begin_time(rngs, frag_intervals):

    begin_times = uniform_begin_time_fn(
        jax.random.split(rngs[0], len(frag_intervals)), frag_intervals
    )
    assert_valid_begin_times(frag_intervals, begin_times)

    begin_times_alt = uniform_begin_time_fn(
        jax.random.split(rngs[1], len(frag_intervals)), frag_intervals
    )
    assert_valid_begin_times(frag_intervals, begin_times_alt)

    assert (begin_times != begin_times_alt).any()


fixed_begin_time_fn = data_fragmentation.get_fixed_begin_time_fn(settings)


@pytest.mark.parametrize(
    "frag_intervals,expected",
    [
        (
            jnp.array([[0, 5], [0, 7], [2, 8], [4, 10]]),
            jnp.array([0.0, 1.0, 2.5, 4.5]),
        )
    ],
)
def test_fixed_begin_time(rngs, frag_intervals, expected):

    begin_times = fixed_begin_time_fn(
        jax.random.split(rngs[0], len(frag_intervals)), frag_intervals
    )
    assert_valid_begin_times(frag_intervals, begin_times)
    assert (begin_times == expected).all()

    begin_times_alt = fixed_begin_time_fn(
        jax.random.split(rngs[1], len(frag_intervals)), frag_intervals
    )

    assert (begin_times == begin_times_alt).all()


pad_tensor_fn = data_fragmentation.get_pad_tensor_fn(settings)


@pytest.mark.parametrize(
    "tensor,begin_times,expected",
    [
        (
            jnp.ones(60),
            jnp.array([-2.0, 1.0, 0.0, 55.0, 57.0]),
            jnp.ones(60 + 6),
        ),
        (
            jnp.ones(30) * 0.5,
            jnp.array([-2.0, 1.0, 0.0, 55.0, 57.0]),
            jnp.concatenate([jnp.ones(30 + 4) * 0.5]),
        ),
    ],
)
def test_pad_tensor(tensor, begin_times, expected):
    padded_tensor, padded_begin_times = pad_tensor_fn(tensor, begin_times)

    pad_size = (padded_tensor.shape[0] - tensor.shape[0]) // 2

    assert (padded_tensor == expected).all()
    assert (padded_begin_times == begin_times + pad_size).all()


slice_fn = data_fragmentation.get_slice_fn(settings)


@pytest.mark.parametrize(
    "tensor,begin_times,expected",
    [
        (
            jnp.arange(60),
            jnp.array([0.0, 12.0, 54.0, 55]),
            jnp.array(
                [
                    [0, 1, 2, 3, 4],
                    [12, 13, 14, 15, 16],
                    [54, 55, 56, 57, 58],
                    [55, 56, 57, 58, 59],
                ]
            ),
        )
    ],
)
def test_slice_fn(tensor, begin_times, expected):
    assert (slice_fn(tensor, begin_times, tensor.shape[0]) == expected).all()


batch_slice_fn = data_fragmentation.get_batch_slice_fn(settings)


@pytest.mark.parametrize(
    "tensor,frag_intervals,expected",
    [
        (
            jnp.arange(60),
            jnp.array([[0.0, 9.0], [2.0, 7.0], [-2.0, 5.0], [57.0, 62.0]]),
            jnp.array(
                [
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [0.0, 0.0, 1.0, 2.0, 3.0],
                    [57.0, 58.0, 59.0, 59.0, 59.0],
                ]
            ),
        )
    ],
)
def test_batch_slice_fn(rng, tensor, frag_intervals, expected):
    slices = batch_slice_fn(rng, tensor, frag_intervals)
    assert (slices == expected).all()
