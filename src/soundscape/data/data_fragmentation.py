import jax
from jax import numpy as jnp
from functools import partial

from ..lib import utils, constants


def valid_begin_interval(fragment_size, interval):
    """
    Get the interval containing all valid begin times of a fragment,
    such that the total length is fragment_size.
    """

    min_val = interval[0]
    max_val = interval[1] - fragment_size
    return min_val, max_val


def uniform_begin_time(settings):
    """
    Sample random begin times from the ranges in frag_interval.
    """

    @jax.jit
    @jax.vmap
    def _begin_time(rng, frag_interval):

        min_begin_time, max_begin_time = valid_begin_interval(
            settings["data"]["fragmentation"]["fragment_size"], frag_interval
        )

        begin_time = jax.random.uniform(
            rng, minval=min_begin_time, maxval=max_begin_time
        )

        return begin_time

    return _begin_time


def fixed_begin_time(settings):
    """
    Return deterministic begin times, which are centered around the fragment intervals
    """

    @jax.jit
    @jax.vmap
    def _begin_time(rng, frag_interval):

        mid_time = (frag_interval[0] + frag_interval[1]) / 2

        begin_time = mid_time - settings["data"]["fragmentation"]["fragment_size"] / 2

        return begin_time

    return _begin_time


begin_time_fns = {
    "uniform": uniform_begin_time,
    "fixed": fixed_begin_time,
}


def pad_tensor(settings):
    """
    Pad a tensor's boundaries such that its fragments can be taken
    from the beginning and end of the tensor without getting out of
    bounds.
    """

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(None, 0))
    def _pad_tensor(tensor, begin_time):

        # Calculates the padding amount
        pad_size = settings["data"]["fragmentation"]["fragment_size"] * (
            1 - settings["data"]["fragmentation"]["min_overlap"]
        )

        # Converts the amount from seconds to samples (the tensor is assumed to represent a 60 seconds clip)
        pad_size_samples = utils.time2pos(tensor.shape[0], pad_size, ceil=True)

        # Updates the begin time variable, since now there is a few additional seconds of padding in the beginning
        begin_time_samples = utils.time2pos(tensor.shape[0], begin_time)
        begin_time_samples = begin_time_samples + pad_size_samples

        # Creates the padding mask
        pad_mask = [(pad_size_samples, pad_size_samples)] + [(0, 0)] * (tensor.ndim - 1)

        # Pads the tensor
        tensor = jnp.pad(
            tensor, pad_mask, settings["data"]["fragmentation"]["padding_mode"]
        )

        return tensor, begin_time_samples

    return _pad_tensor


def slice(settings):
    """
    Return a slice of the given tensor starting from begin_pos with a length of fragment_size.

    Args:
        tensor: a tensor to be sliced
        begin_pos: the position to start the slice
        fragment_size: the length of the slice
    """

    @partial(jax.vmap, in_axes=(None, 0, None))
    def _slice(tensor, begin_pos, fragment_size):

        start_indices = [begin_pos] + [0] * (tensor.ndim - 1)
        slice_sizes = [fragment_size] + list(tensor.shape[1:])

        return jax.lax.dynamic_slice(tensor, start_indices, slice_sizes)

    return _slice


def slice_fragments(settings):
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

    _begin_time_fn = begin_time_fns[settings["data"]["fragmentation"]["begin_time_fn"]](
        settings
    )
    _pad_tensor = pad_tensor(settings)
    _slice = slice(settings)

    def _slice_fragments(rng, tensor, frag_intervals):
        # Finds the sampling function used for the begin times

        # Samples the begin times from the given intervals
        rngs = jax.random.split(rng, frag_intervals.shape[0])
        begin_times = _begin_time_fn(rngs, frag_intervals)

        # Pads the tensor
        padded_tensor, padded_begin_pos = _pad_tensor(tensor, begin_times)

        fragment_size = utils.time2pos(
            tensor.shape[0], settings["data"]["fragmentation"]["fragment_size"]
        )

        # Slices the padded tensor into fragments
        return _slice(padded_tensor, padded_begin_pos, fragment_size)

    return _slice_fragments


def dict_slice_fragments(settings):
    """
    Applies the slice_fragments function to a dict of arguments.
    The dict must contain the keys "rng", "frag_intervals" and at
    least one of "spec" or "wav".
    """

    _slice_fragments = slice_fragments(settings)

    @utils.jax2tf_fn
    def _dict_slice_fragments(args):

        # Crops the slices of the spectrogram and waves
        for key in {"spec", "wav"}.intersection(set(args.keys())):
            args[key] = _slice_fragments(args["rng"], args[key], args["frag_intervals"])

        # Removes padded slices from the dict

        num_events = args["num_events"]

        for key in args:
            if type(args[key]) == str:
                args[key] = [args[key]] * int(num_events)
            elif args[key].ndim >= 1 and args[key].shape[0] == constants.MAX_EVENTS:
                args[key] = args[key][:num_events]
            else:
                args[key] = jnp.repeat(args[key][None, ...], num_events, axis=0)

        return args

    return _dict_slice_fragments
