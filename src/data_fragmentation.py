from random import uniform
import tensorflow as tf
import jax
from jax import numpy as jnp
from functools import partial

import utils
import constants


def get_uniform_begin_time_fn(settings):
    """
    Sample a random time from the range [minimum_time, max_time].
    """

    @jax.vmap
    def uniform_begin_time(rng, frag_interval):

        minimum_begin_time = frag_interval[0]
        maximum_begin_time = frag_interval[1] - settings["fragment_size"]

        begin_time = jax.random.uniform(
            rng, minval=minimum_begin_time, maxval=maximum_begin_time
        )
        return begin_time

    return uniform_begin_time


def get_fixed_begin_time_fn(settings):
    """
    Returns a fixed time, which is centered around the fragment interval
    """

    @jax.vmap
    def fixed_begin_time(rng, frag_interval):

        frag_interval_middle = (frag_interval[0] + frag_interval[1]) / 2
        begin_time = frag_interval_middle - settings["fragment_size"] / 2

        return begin_time

    return fixed_begin_time


begin_time_fns = {
    "uniform": get_uniform_begin_time_fn,
    "fixed": get_fixed_begin_time_fn,
}


def get_pad_tensor_fn(settings):
    """
    Pad a tensor's boundaries such that its fragments can be taken
    from the beginning and end of the tensor without getting out of
    bounds.
    """

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(None, 0))
    def pad_tensor(tensor, begin_time):

        pad_size = utils.time2pos(
            tensor.shape[0],
            settings["fragment_size"] * (1 - settings["min_overlap"]),
            ceil=True,
        )

        pad_mask = jnp.concatenate(
            [
                jnp.array([[pad_size, pad_size]], dtype=jnp.int32),
                jnp.zeros((len(tensor.shape) - 1, 2), dtype=jnp.int32),
            ],
            axis=0,
        )

        tensor = jnp.pad(tensor, pad_mask, settings["padding_mode"])

        begin_time = begin_time + pad_size

        return tensor, begin_time

    return pad_tensor


def get_slice_fn(settings):
    """
    Return a slice of the given tensor starting from begin_time with a length of settings["fragment_size"].
    """

    @partial(jax.vmap, in_axes=(None, 0, None))
    def slice(tensor, begin_time, valid_length):

        begin_pos = utils.time2pos(valid_length, begin_time)

        fragment_size = utils.time2pos(valid_length, settings["fragment_size"])

        one_mask = jnp.array([1] + [0] * (len(tensor.shape) - 1), dtype=jnp.int32)
        shape_mask = jnp.array(tensor.shape) * (1 - one_mask)

        return jax.lax.dynamic_slice(
            tensor,
            begin_pos * one_mask,
            (fragment_size) * one_mask + shape_mask,
        )

    return slice


def get_batch_slice_fn(settings):
    """
    Using the functions above, create a function that slices a
    tensor according to the fragments in frag_interval.

    Args:
        rng: a jax random seed
        tensor: a tensor to be sliced
        frag_intervals: a numpy array of shape (n_fragments, 2) containing the minimum begin and maximum end times of the fragments

    Returns:
        A tensor of shape (n_fragments, fragment_size) containing the fragments sliced from the tensor.
    """

    begin_time_fn = begin_time_fns[settings["begin_time_fn"]](settings)
    pad_tensor_fn = get_pad_tensor_fn(settings)
    slice_fn = get_slice_fn(settings)

    def batch_slice(rng, tensor, frag_intervals):
        rngs = jax.random.split(rng, frag_intervals.shape[0])
        begin_times = begin_time_fn(rngs, frag_intervals)
        padded_tensor, padded_begin_times = pad_tensor_fn(tensor, begin_times)
        return slice_fn(padded_tensor, padded_begin_times, tensor.shape[0])

    return batch_slice
