from random import uniform

import jax
import pytest
from jax import numpy as jnp
import tensorflow as tf
import numpy as np

import data_fragmentation
import utils

from settings import settings


@pytest.mark.parametrize(
    "fragment_size,interval,expected",
    [
        (1, (0, 10), (0, 9)),
        (2, (2, 8), (2, 6)),
        (0, (1, 5), (1, 5)),
    ],
)
def test_valid_begin_interval(fragment_size, interval, expected):
    assert data_fragmentation.valid_begin_interval(fragment_size, interval) == expected


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def rngs(rng):
    return jax.random.split(rng, 2)


def assert_valid_begin_times(frag_intervals, begin_times):
    assert begin_times.shape == (len(frag_intervals),)
    assert (begin_times >= frag_intervals[:, 0]).all()
    assert (
        begin_times + settings["data"]["fragmentation"]["fragment_size"]
        <= frag_intervals[:, 1]
    ).all()


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


def test_fragmentation_fn():

    _ = data_fragmentation.get_jax_fragmentation_fn(settings)
    jax_process_data, flatten_inputs_fn, unflatten_outputs_fn, output_types = _

    example_instance = {
        "filename": "/path/to/file",
        "time_intervals": [[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [0.0, 0.0]],
        "freq_intervals": [[440.0, 880.0], [440.0, 880.0], [440.0, 880.0], [0.0, 0.0]],
        "frag_intervals": [[1.0, 10.0], [2.0, 50.0], [3.0, 60.0], [0.0, 0.0]],
        "labels": [0, 1, 2, -1],
        "num_events": 3,
        "rng": np.array(jax.random.PRNGKey(0)),
        "spec": np.zeros((256, 6000), dtype=np.uint16),
    }

    example_instance = {k: tf.constant(v) for k, v in example_instance.items()}

    flattened_instance = flatten_inputs_fn(example_instance)
    flattened_fragments = jax_process_data(*flattened_instance)
    fragments = unflatten_outputs_fn(flattened_fragments)

    n = example_instance["num_events"]
    assert tf.math.reduce_all(
        fragments["filename"] == tf.repeat(example_instance["filename"], n)
    )
    assert tf.math.reduce_all(
        fragments["time_intervals"] == example_instance["time_intervals"][0:n]
    )
    assert fragments["slice"].shape == (
        3,
        100 * settings["data"]["fragmentation"]["fragment_size"],
        256,
    )

    flattened_dtypes = [x.dtype for x in flattened_fragments]

    for d1, d2 in zip(flattened_dtypes, output_types):
        assert d1 == d2
