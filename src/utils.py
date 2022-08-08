from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tensorflow as tf
import jax
import jax.dlpack
from jax import numpy as jnp


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


def tf2jax(x):
    """
    Convert a tensor to a jax.numpy array (or pytree).
    """

    def f(x):
        if x.dtype != tf.string:
            x = tf.experimental.dlpack.to_dlpack(x)
            x = jax.dlpack.from_dlpack(x)
        else:
            x = x.numpy()
        return x

    return jax.tree_map(f, x)


def jax2tf(x):
    """
    Convert a jax.numpy array (or pytree) to a tensor.
    """

    def f(x):
        x = jax.dlpack.to_dlpack(x)
        x = tf.experimental.dlpack.from_dlpack(x)
        return x

    return jax.tree_map(f, x)


def time2pos(tensor_length, time, total_duration=60, ceil=False):
    """
    Convert a time to a position in a tensor.
    """
    x = tensor_length * time / total_duration
    if ceil:
        x = jnp.ceil(x)
    return jnp.int32(x)


class hash_dict(dict):
    """
    A dictionary that is hashable
    """

    def __hash__(self):
        return hash(tuple(sorted(self.items())))
