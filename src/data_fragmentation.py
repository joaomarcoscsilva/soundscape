import jax
from jax import numpy as jnp
from functools import partial

import utils
from settings import settings
import constants

frag_settings = settings["data"]["fragmentation"]


def valid_begin_interval(fragment_size, interval):
    """
    Get the interval containing all valid begin times of a fragment,
    such that the total length is fragment_size.
    """

    min_val = interval[0]
    max_val = interval[1] - fragment_size
    return min_val, max_val


@jax.jit
@jax.vmap
def uniform_begin_time(rng, frag_interval):
    """
    Sample random begin times from the ranges in frag_interval.
    """

    min_begin_time, max_begin_time = valid_begin_interval(
        frag_settings["fragment_size"], frag_interval
    )

    begin_time = jax.random.uniform(rng, minval=min_begin_time, maxval=max_begin_time)

    return begin_time


@jax.jit
@jax.vmap
def fixed_begin_time(rng, frag_interval):
    """
    Return deterministic begin times, which are centered around the fragment intervals
    """

    mid_begin_time = (frag_interval[0] + frag_interval[1]) / 2

    begin_time = mid_begin_time - frag_settings["fragment_size"] / 2

    return begin_time


begin_time_fns = {
    "uniform": uniform_begin_time,
    "fixed": fixed_begin_time,
}


@partial(jax.vmap, in_axes=(None, 0), out_axes=(None, 0))
def pad_tensor(tensor, begin_time):
    """
    Pad a tensor's boundaries such that its fragments can be taken
    from the beginning and end of the tensor without getting out of
    bounds.
    """

    # Calculates the padding amount
    pad_size = frag_settings["fragment_size"] * (1 - frag_settings["min_overlap"])

    # Converts the amount from seconds to samples (the tensor is assumed to represent a 60 seconds clip)
    pad_size = utils.time2pos(pad_size, tensor.shape[0], ceil=True)

    # Creates the padding mask
    pad_mask = [(pad_size, pad_size)] + [(0, 0)] * (tensor.ndim - 1)

    # Pads the tensor
    tensor = jnp.pad(tensor, pad_mask, frag_settings["padding_mode"])

    # Updates the begin time variable, since now there is a few additional seconds of padding in the beginning
    begin_time = begin_time + pad_size

    return tensor, begin_time


@partial(jax.vmap, in_axes=(None, 0, None))
def slice(tensor, begin_time, valid_length=None):
    """
    Return a slice of the given tensor starting from begin_time with a length of settings["data"]["fragmentation"]["fragment_size"].

    Args:
        tensor: a tensor to be sliced
        begin_time: the time at which the slice should start
        valid_length: if the tensor is padded, the length of the valid part of the tensor (i.e. the length of the original tensor)
    """

    if valid_length is None:
        valid_length = tensor.shape[0]

    # Converts the arguments from seconds to samples (the tensor is assumed to represent a 60 seconds clip)
    begin_pos = utils.time2pos(begin_time, valid_length)
    fragment_size = utils.time2pos(frag_settings["fragment_size"], valid_length)

    #
    start_indices = [begin_pos] + [0] * (tensor.ndim - 1)
    slice_sizes = [fragment_size] + list(tensor.shape[1:])

    return jax.lax.dynamic_slice(tensor, start_indices, slice_sizes)


def slice_fragments(rng, tensor, frag_intervals):
    """
    Return a list of fragments from the given tensor, sampled from the given intervals.

    This function samples fragments from the given intervals, pads the tensor and then slices it into fragments.

    Args:
        rng: a jax random seed
        tensor: a tensor to be sliced
        frag_intervals: a numpy array of shape (n_fragments, 2) containing the minimum begin and maximum end times of the fragments

    Returns:
        A tensor of shape (n_fragments, fragment_size) containing the fragments sliced from the tensor.
    """

    # Finds the sampling function used for the begin times
    begin_time_fn = begin_time_fns[frag_settings["begin_time_fn"]]

    # Samples the begin times from the given intervals
    rngs = jax.random.split(rng, frag_intervals.shape[0])
    begin_times = begin_time_fn(rngs, frag_intervals)

    # Pads the tensor
    padded_tensor, padded_begin_times = pad_tensor(tensor, begin_times)

    # Slices the padded tensor into fragments
    return slice(padded_tensor, padded_begin_times, tensor.shape[0])


@utils.jax2tf_fn
def dict_slice_fragments(args):
    """
    Applies the slice_fragments function to a dict of arguments.
    The dict must contain the keys "rng", "frag_intervals" and at
    least one of "spec" or "wav".
    """

    # Crops the slices of the spectrogram and waves
    for key in {"spec", "wav"}.intersection(set(args.keys())):
        args[key] = slice_fragments(args["rng"], args[key], args["frag_intervals"])

    # Removes padded slices from the dict

    num_events = args["num_events"]

    for key in args:
        if type(args[key]) == str:
            args[key] = [args[key]] * num_events
        elif args[key].ndim >= 1 and args[key].shape[0] == constants.MAX_EVENTS:
            args[key] = args[key][:num_events]
        else:
            args[key] = jnp.repeat(args[key][None, ...], num_events, axis=0)

    return args
