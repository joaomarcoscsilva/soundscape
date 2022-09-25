from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tensorflow as tf
import jax
import jax.dlpack
from jax import numpy as jnp
from functools import wraps
from typing import Callable
import copy


def dict_map(fn, d):
    """
    Map a function over a dictionary.
    """

    return {k: fn(v) for k, v in d.items()}


def parallel_map(fn, args, use_tqdm=True):
    """
    Map a function to a list of arguments in parallel.
    Note that fn must be pickleable.
    """

    with Pool(cpu_count()) as p:
        if use_tqdm:
            return list(
                tqdm(
                    p.imap(fn, args),
                    total=len(args) if args.__len__ else None,
                )
            )
        else:
            return list(p.imap(fn, args))


def tf2jax(*args, to_gpu=False):
    """
    Convert a tensor to a jax.numpy array (or pytree).
    """

    def f(x):
        if x.dtype != tf.string:
            x = tf.experimental.dlpack.to_dlpack(x)
            x = jax.dlpack.from_dlpack(x)
            if to_gpu:
                x = jax.device_put(x, jax.devices()[0])
        else:
            x = x.numpy()
            if type(x) == bytes:
                x = x.decode()
            else:
                x = list(map(lambda x: x.decode(), x))

        return x

    if len(args) > 1:
        return [jax.tree_util.tree_map(f, x) for x in args]

    return jax.tree_util.tree_map(f, args[0])


def jax2tf(*args):
    """
    Convert a jax.numpy array (or pytree) to a tensor.
    """

    def f(x):
        if type(x) not in {str, bytes}:
            # x = jax.dlpack.to_dlpack(x)
            # x = tf.experimental.dlpack.from_dlpack(x)
            x = tf.constant(x)
        else:
            x = tf.constant(x)
        return x

    if len(args) > 1:
        return [jax.tree_util.tree_map(f, x) for x in args]

    return jax.tree_util.tree_map(f, args[0])


def jax2tf_fn(fn):
    """
    Converts a function that works on jax.numpy arrays to one that works on tensors.
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        return jax2tf(fn(*tf2jax(args), **tf2jax(kwargs)))

    return wrapped


def tf2jax_fn(fn):
    """
    Converts a function that works on tensors to one that works on jax.numpy arrays.
    """

    @wraps(fn)
    def wrapped_fn(*args, **kwags):
        return tf2jax(fn(*jax2tf(args), **jax2tf(kwags)))

    return wrapped_fn


def time2pos(tensor_length, time, ceil=False):
    """
    Convert a time to a position in a tensor.
    """
    x = tensor_length * time / 60
    if ceil:
        x = jnp.ceil(x)
    return jnp.int32(x)


class hash_dict(dict):
    """
    A dictionary that is hashable
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = hash_dict(v)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def copy(self):
        return copy.deepcopy(self)


def flatten_fn(fn: Callable[[dict], dict], input_keys: list, output_keys: list):
    """
    Turn a function that accepts a dictionary into one that accepts positional arguments.
    Similarly, results are returned as a list instead of a dictionary.

    Args:
        fn: a jax function that recieves and returns a dict
        input_keys: a set of the keys of the input dict
        output_types: a dict containing the types of the values in the returned dict
    """

    @wraps(fn)
    def wrapped_fn(*args):
        args_dict = dict(zip(input_keys, args))
        results_dict = fn(args_dict)
        return [results_dict[key] for key in output_keys]

    return wrapped_fn


def unflatten_fn(flat_fn: Callable, input_keys: list, output_keys: list):
    """
    Turn a function that accepts positional arguments into one that accepts a dictionary.
    Similarly, results are returned as a dictionary instead of a list.

    Args:
        fn: a jax function that recieves positional arguments and returns a list
        input_keys: a set of the keys of the input dict
        output_types: a dict containing the types of the values in the returned dict
    """

    @wraps(flat_fn)
    def wrapped_fn(args):
        flat_results = flat_fn(*[args[key] for key in input_keys])
        results_dict = dict(zip(output_keys, flat_results))
        return results_dict

    return wrapped_fn


def tf_py_func_flat(flat_fn: Callable, output_types: list):
    """
    Wrap a flat function (positional arguments only) around a tf.py_func call.
    """

    @wraps(flat_fn)
    def wrapped_fn(*flattened_args):
        return tf.py_function(flat_fn, flattened_args, output_types)

    return wrapped_fn


def tf_py_func(fn: Callable[[dict], dict], input_keys: list, output_types: dict):
    """
    Wrap a non-flat (dict argument) function around a tf.py_func call.
    """

    flat_fn = flatten_fn(fn, input_keys, list(output_types.keys()))
    tf_func = tf_py_func_flat(flat_fn, list(output_types.values()))
    return unflatten_fn(tf_func, input_keys, list(output_types.keys()))


def wrap_around(fn, outer_fn):
    """
    Create a function that calls fn and then passes the result to outer_fn.
    All inputs passed to fn are also passed to outer_fn.
    """

    def _fn(*args, **kwargs):
        return outer_fn(fn(*args, **kwargs), *args, **kwargs)

    return _fn


def remove_index(tup, index):
    """
    Remove an index from a tuple.
    """
    return tuple(v for i, v in enumerate(tup) if i != index), tup[index]


def insert_index(tup, index, val):
    if index < 0:
        index += len(tup) + 1
    return tup[:index] + (val,) + tup[index:]


def remove_key(dic, key):
    return {k: v for k, v in dic.items() if k != key}, dic[key]
