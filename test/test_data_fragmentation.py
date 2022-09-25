import jax
import pytest
from jax import numpy as jnp
import tensorflow as tf
import numpy as np

from soundscape.data import data_fragmentation
from soundscape.lib import constants
from soundscape.lib.settings import settings


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


@pytest.mark.parametrize(
    "frag_intervals", [jnp.array([[0, 5], [0, 7], [2, 8], [4, 10]])]
)
def test_uniform_begin_time(rngs, frag_intervals):

    begin_times = data_fragmentation.uniform_begin_time(
        jax.random.split(rngs[0], len(frag_intervals)), frag_intervals
    )
    assert_valid_begin_times(frag_intervals, begin_times)

    begin_times_alt = data_fragmentation.uniform_begin_time(
        jax.random.split(rngs[1], len(frag_intervals)), frag_intervals
    )
    assert_valid_begin_times(frag_intervals, begin_times_alt)

    assert (begin_times != begin_times_alt).any()


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

    begin_times = data_fragmentation.fixed_begin_time(
        jax.random.split(rngs[0], len(frag_intervals)), frag_intervals
    )
    assert_valid_begin_times(frag_intervals, begin_times)
    assert (begin_times == expected).all()

    begin_times_alt = data_fragmentation.fixed_begin_time(
        jax.random.split(rngs[1], len(frag_intervals)), frag_intervals
    )

    assert (begin_times == begin_times_alt).all()


@pytest.mark.parametrize(
    "tensor,begin_times,expected_tensor,expected_pos",
    [
        (
            jnp.ones(60),
            jnp.array([-2.0, 1.0, 0.0, 55.0, 57.0]),
            jnp.ones(60 + 6),
            jnp.array([1, 4, 3, 58, 60]),
        ),
        (
            jnp.ones(30) * 0.5,
            jnp.array([-2.0, 1.0, 0.0, 55.0, 57.0]),
            jnp.ones(30 + 4) * 0.5,
            jnp.array([1, 2, 2, 29, 30]),
        ),
    ],
)
def test_pad_tensor(tensor, begin_times, expected_tensor, expected_pos):
    padded_tensor, padded_pos = data_fragmentation.pad_tensor(tensor, begin_times)
    assert (padded_tensor == expected_tensor).all()
    assert (padded_pos == expected_pos).all()


@pytest.mark.parametrize(
    "tensor,begin_pos,fragment_size,expected",
    [
        (
            jnp.arange(60),
            jnp.array([0, 12, 54, 55]),
            5,
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
def test_slice(tensor, begin_pos, fragment_size, expected):
    slices = data_fragmentation.slice(tensor, begin_pos, fragment_size)
    assert (slices == expected).all()


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
def test_slice_fragments(rng, tensor, frag_intervals, expected):
    local_settings = {**settings}
    local_settings["data"]["fragmentation"]["begin_time_fn"] = "fixed"

    slices = data_fragmentation.slice_fragments.call(
        local_settings, rng, tensor, frag_intervals
    )

    assert (slices == expected).all()


def test_dict_slice_fragments():

    test_instance = {
        # Necessary keys:
        "rng": tf.constant(jax.random.PRNGKey(0)),
        "frag_intervals": tf.zeros((constants.MAX_EVENTS, 2)),
        "spec": tf.zeros((6000, 256), dtype=tf.uint16),
        "num_events": tf.constant(5),
        # Test keys:
        "string": tf.constant("string_here"),
        "non_padded_array": tf.zeros(7),
        "padded_array": tf.zeros(constants.MAX_EVENTS),
        "padded_tensor": tf.zeros((constants.MAX_EVENTS, 3, 5)),
        "integer": tf.constant(4),
    }

    output = data_fragmentation.dict_slice_fragments(test_instance)

    for k, v in output.items():

        assert len(v) == test_instance["num_events"]

        if test_instance[k].ndim > 0:
            if k == "spec":
                fragment_size = settings["data"]["fragmentation"]["fragment_size"]
                assert v.shape[1:] == (
                    len(test_instance["spec"]) * fragment_size // 60,
                    settings["data"]["spectrogram"]["n_mels"],
                )
            elif test_instance[k].dtype == tf.string:
                assert v[0] == v[-1] == test_instance[k]
            elif test_instance[k].shape[0] == constants.MAX_EVENTS:
                assert v.shape[1:] == test_instance[k].shape[1:]
            else:
                assert v.shape[1:] == test_instance[k].shape
