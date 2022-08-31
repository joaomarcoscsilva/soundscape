import utils
import tensorflow as tf
from jax import numpy as jnp
import jax

import pytest


# Functions used in test_parallel_map since it's not possible to use lambdas.
def inc_fn(x):
    return x + 1


def sqr_fn(x):
    return x * x


@pytest.mark.parametrize(
    "fn,x",
    [
        (inc_fn, [1, 2, 3, 4, 5]),
        (inc_fn, []),
        (sqr_fn, [-1, 0, 1, 2, 3]),
    ],
)
def test_parallel_map(fn, x):
    y_pred = utils.parallel_map(fn, x, use_tqdm=False)
    y_true = list(map(fn, x))
    assert y_pred == y_true


@pytest.mark.parametrize(
    "array",
    [
        [1, 2, 3, 4, 5],
        [1],
        {"a": [1, 2], "b": [3, 4], "c": [[5], [6]], "d": "string"},
        "string",
        ["string1", "string2"],
    ],
)
def test_tf_jax(array):
    if type(array) == list:
        if type(array[0]) != str:
            x_tf = tf.constant(array)
            x_jax = jnp.array(array)
        else:
            x_jax = array
            x_tf = tf.constant(array)
    else:
        x_tf = jax.tree_util.tree_map(tf.constant, array)
        x_jax = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if type(x) != str else x, array
        )

    y_jax = utils.tf2jax(x_tf)
    y_tf = utils.jax2tf(x_jax)

    assert jnp.all(y_jax == x_jax)
    assert jnp.all(utils.tf2jax(y_tf) == x_jax)

    y_jax1, y_jax2 = utils.tf2jax(x_tf, x_tf)
    y_tf1, y_tf2 = utils.jax2tf(x_jax, x_jax)

    assert jnp.all(y_jax1 == x_jax)
    assert jnp.all(y_jax2 == x_jax)
    assert jnp.all(utils.tf2jax(y_tf1) == x_jax)
    assert jnp.all(utils.tf2jax(y_tf2) == x_jax)


def test_tf_jax_fn():
    jax_fn = lambda x, y, z: jnp.mean(x + y + z)
    tf_fn = lambda x, y, z: tf.reduce_mean(x + y + z)

    x = [1, 2, 3]
    y = [2, 3, 4]
    z = [3, 4, 5]

    jax_x = jnp.array(x)
    jax_y = jnp.array(y)
    jax_z = jnp.array(z)

    tf_x = tf.constant(x)
    tf_y = tf.constant(y)
    tf_z = tf.constant(z)

    assert jax_fn(jax_x, jax_y, z=jax_z) == 9
    assert tf_fn(tf_x, tf_y, z=tf_z) == 9

    assert utils.jax2tf_fn(jax_fn)(tf_x, tf_y, z=tf_z) == 9
    assert utils.tf2jax_fn(tf_fn)(jax_x, jax_y, z=jax_z) == 9


@pytest.mark.parametrize(
    "tensor_length, time, expected",
    [
        (60, 0, 0),
        (60, 5, 5),
        (60, 61, 61),
        (60, -1, -1),
        (128, 30, 64),
        (128, 15, 32),
        (10, 12, 2),
    ],
)
def test_time2pos(tensor_length, time, expected):
    result = utils.time2pos(time, tensor_length)
    assert result == expected


def assert_dicts_equal(d1, d2):
    assert d1.keys() == d2.keys()
    for key in d1.keys():
        if type(d1[key]) == dict:
            assert_dicts_equal(d1[key], d2[key])
        else:
            assert d1[key] == d2[key]


@pytest.mark.parametrize(
    "d",
    [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": {"c": 5, "d": 6}, "b": 2},
    ],
)
def test_hash_dict(d):
    d_hashed = utils.hash_dict(d)
    assert_dicts_equal(d, d_hashed)
    assert hash(d_hashed) is not None


def test_flatten_unflatten_fn():

    flat_fn = lambda a, b: [a, a + b, b]
    unflat_fn = lambda args: {"c": args["a"] + args["b"], **args}

    input_keys = ["a", "b"]
    output_keys = ["a", "c", "b"]

    dict_inputs = {"a": 1, "b": 2}
    dict_outputs = {"a": 1, "b": 2, "c": 3}
    list_inputs = [1, 2]
    list_outputs = [1, 3, 2]

    # Test that flattening and unflattening work
    f_flattened = utils.flatten_fn(unflat_fn, input_keys, output_keys)
    assert f_flattened(*list_inputs) == list_outputs

    f_unflattened = utils.unflatten_fn(flat_fn, input_keys, output_keys)
    assert f_unflattened(dict_inputs) == dict_outputs

    # Test that they are inverses of each other
    f_flattened_unflattened = utils.flatten_fn(f_unflattened, input_keys, output_keys)
    assert f_flattened_unflattened(*list_inputs) == list_outputs

    f_unflattened_flattened = utils.unflatten_fn(f_flattened, input_keys, output_keys)
    assert f_unflattened_flattened(dict_inputs) == dict_outputs


def test_tf_py_func_flat():

    flat_fn = lambda a, b: [a, tf.cast(a + b, tf.float32), b]
    output_types = [tf.int32, tf.float32, tf.int32]

    f = utils.tf_py_func_flat(flat_fn, output_types)

    x = [tf.constant(1), tf.constant(2)]
    expected = [1, 3, 2]

    for i, y in enumerate(f(*x)):
        assert y.dtype == output_types[i]
        assert y.numpy() == expected[i]


def test_tf_py_func():

    unflat_fn = lambda args: {"c": tf.cast(args["a"] + args["b"], tf.float32), **args}
    input_keys = ["a", "b"]
    output_types = {"a": tf.int32, "b": tf.int32, "c": tf.float32}

    f = utils.tf_py_func(unflat_fn, input_keys, output_types)

    x = {"a": tf.constant(1), "b": tf.constant(2)}
    expected = {"a": 1, "b": 2, "c": 3}

    for k, y in f(x).items():
        assert y.dtype == output_types[k]
        assert y.numpy() == expected[k]
