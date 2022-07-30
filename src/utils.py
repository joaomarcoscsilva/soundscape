from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tensorflow as tf
import jax
from jax import numpy as jnp


def parallel_map(fn, args):
    """
    Map a function to a list of arguments in parallel.
    """

    with Pool(cpu_count()) as p:
        return list(tqdm(p.imap(fn, args), total=len(args) if args.__len__ else None))


def tf2jax(x):
    """
    Convert a tensor to a jax.numpy array.
    """

    x = tf.experimental.dlpack.to_dlpack(x)
    x = jax.dlpack.from_dlpack(x)

    return x


def jax2tf(x):
    """
    Convert a jax.numpy array to a tensor.
    """

    x = jax.dlpack.to_dlpack(x)
    x = tf.experimental.dlpack.from_dlpack(x)

    return x
