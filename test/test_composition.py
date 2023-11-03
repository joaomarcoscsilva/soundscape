from functools import partial

import jax
import pytest
from jax import numpy as jnp

from soundscape.composition import *


def test_state_addition_with_repeated_keys():
    state1 = State({"a": 1, "b": 2})
    state2 = State({"b": 3, "c": 4})
    added = state1 + state2
    assert added == State({"a": 1, "b": 3, "c": 4})


def test_state_addition_from_empty_state():
    state1 = State({"a": 1, "b": 2})
    state2 = State()
    added = state1 + state2
    assert added == State({"a": 1, "b": 2})


def test_state_key_selection():
    state = State({"a": 1, "b": 2, "c": 3})
    substate = state.select_keys(["a", "c"])
    assert substate == State({"a": 1, "c": 3})


def test_state_key_selection_with_mapping():
    state = State({"a": 1, "b": 2, "c": 3})
    substate = state.select_keys(["a", "x"], key_map={"x": "c"})
    assert substate == State({"a": 1, "x": 3})


def test_state_is_hashable():
    state1 = State({"a": 1, "b": 2})
    state2 = State({"b": 2, "a": 1})
    state3 = State({"a": 1, "b": 3})

    assert hash(state1) == hash(state2)
    assert hash(state1) != hash(state3)


def test_state_split():
    state = State({"a": 1, "b": 2, "c": 3, "d": 4})
    state1, state2 = state.split(["a", "c"])

    assert state1 == State({"a": 1, "c": 3})
    assert state2 == State({"b": 2, "d": 4})


def test_state_split_with_empty_keys():
    state = State({"a": 1, "b": 2, "c": 3, "d": 4})
    state1, state2 = state.split([])

    assert state1 == State({})
    assert state2 == State({"a": 1, "b": 2, "c": 3, "d": 4})


def test_state_is_pytree():
    state = State({"a": 1, "b": 2, "c": 3, "d": 4})
    leaves, treedef = jax.tree_util.tree_flatten(state)
    reconstructed_state = jax.tree_util.tree_unflatten(treedef, leaves)
    assert state == reconstructed_state


def add_fn(x):
    return x + 1


def mul_fn(x):
    return x * 2


def test_composable_for_simple_functions():
    f = Composable(add_fn)
    g = Composable(mul_fn)
    fn = f | g

    assert fn(1) == 4
    assert fn(2) == 6
    assert fn(0.5) == 3


def test_composable_for_list_of_functions():
    fn = Composable([add_fn, add_fn, mul_fn])

    assert fn(1) == 6
    assert fn(2) == 8
    assert fn(0.5) == 5


def test_composable_is_idempotent():
    f1 = Composable(add_fn)
    f2 = Composable(f1)

    assert f1(1) == f2(1) == 2
    assert f1(2) == f2(2) == 3
    assert f1(0.5) == f2(0.5) == 1.5


def test_composable_identity():
    f1 = Composable(add_fn)
    f2 = identity

    fn1 = f1 | f2
    fn2 = f2 | f1

    assert fn1(1) == fn2(1) == 2
    assert fn1(2) == fn2(2) == 3
    assert fn1(0.5) == fn2(0.5) == 1.5


def test_composable_is_associative():
    f1 = Composable(add_fn)
    f2 = Composable(mul_fn)
    f3 = Composable(add_fn)

    fn1 = (f1 | f2) | f3
    fn2 = f1 | (f2 | f3)

    assert fn1(1) == fn2(1) == 5
    assert fn1(2) == fn2(2) == 7
    assert fn1(0.5) == fn2(0.5) == 4


def test_collapse_all_functions():
    f1 = Composable([add_fn, add_fn, mul_fn])
    f2 = f1._collapse()

    assert f1(1) == f2(1) == 6
    assert f1(2) == f2(2) == 8
    assert f1(0.5) == f2(0.5) == 5


def test_collapse_traceable_functions():
    f1 = Composable(add_fn)
    f2 = Composable.untraceable(mul_fn)

    fn = f1 | f1 | f2 | f1

    fn_collapsed = fn._collapse_traceable()

    assert fn(1) == fn_collapsed(1) == 7
    assert fn(2) == fn_collapsed(2) == 9
    assert fn(0.5) == fn_collapsed(0.5) == 6

    assert len(fn_collapsed.fns) == 3

    assert fn.traceable == [True, True, False, True]
    assert fn_collapsed.traceable == [True, False, True]


def test_state_function_simple_creation():
    f = StateFunction(lambda x, y: x + y).output("out")
    inputs = State({"x": 1, "y": 2})
    outputs = f(inputs)
    assert outputs == State({"x": 1, "y": 2, "out": 3})


def test_state_function_creation_with_state_output():
    f = StateFunction(lambda x, y: State({"out": x + y, "sub": x - y}))
    inputs = State({"x": 1, "y": 2})
    outputs = f(inputs)
    assert outputs == State({"x": 1, "y": 2, "out": 3, "sub": -1})


def test_state_function_creation_with_mapped_input():
    f = StateFunction(lambda x, y: x - y).output("out").inputs({"x": "a", "y": "b"})
    inputs = State({"a": 1, "b": 2})
    outputs = f(inputs)
    assert outputs == State({"a": 1, "b": 2, "out": -1})


def test_jit_a_single_function():
    @StateFunction.with_output("out")
    def fn(static, x):
        return static + x

    jitted_fn = Composable._jit_fn(fn, static_keys=["static"], ignored_keys=["ignored"])

    state = State({"x": 1, "static": 2, "ignored": 3, "_underlined": 4})

    outs = fn(state)
    jitted_outs = jitted_fn(state)

    assert (
        outs
        == jitted_outs
        == {"x": 1, "out": 3, "static": 2, "ignored": 3, "_underlined": 4}
    )

    for key in outs:
        assert not isinstance(outs[key], jnp.ndarray)

    keys_that_should_be_jnp_arrays = ["x", "out"]
    for key in jitted_outs:
        if key in keys_that_should_be_jnp_arrays:
            assert isinstance(jitted_outs[key], jnp.ndarray)
        else:
            assert not isinstance(jitted_outs[key], jnp.ndarray)


def test_jit_function_chain():
    @StateFunction.with_output("x")
    def fn1(x, static):
        return x + static

    @StateFunction.with_output("x")
    def fn2(x, static):
        return x - static

    @StateFunction.with_output("y", traceable=False)
    def untraceable_fn(x):
        return str(x)

    fn = fn1 | fn2 | fn1 | untraceable_fn | fn1
    jitted_fn = fn.jit(static_keys=["static"], ignored_keys=["y"])

    assert jitted_fn.fns

    inputs = State({"x": 1, "static": 2})

    outs = fn(inputs)
    jitted_outs = jitted_fn(inputs)

    assert outs == jitted_outs == {"x": 5, "y": "3", "static": 2}

    for key in outs:
        assert not isinstance(outs[key], jnp.ndarray)

    keys_that_should_be_jnp_arrays = ["x"]
    for key in jitted_outs:
        if key in keys_that_should_be_jnp_arrays:
            assert isinstance(jitted_outs[key], jnp.ndarray)
        else:
            assert not isinstance(jitted_outs[key], jnp.ndarray)


def test_vmap_fails_for_untraceable_function():
    @StateFunction.with_output("y")
    def fn1(x):
        return x + 1

    @StateFunction.with_output("z")
    def fn2(y):
        return y * 2

    @StateFunction.with_output("w", traceable=False)
    def untraceable_fn(l):
        return str(l)

    fn = fn1 | fn2 | untraceable_fn
    with pytest.raises(ValueError):
        fn.vmap(["x"], ["y", "z"])


def test_vmap_works_for_traceable_functions():
    @StateFunction.with_output("y")
    def fn1(x):
        return x + 1

    @StateFunction.with_output("z")
    def fn2(y):
        return y * 2

    fn = fn1 | fn2
    vmap_fn = fn.vmap(["x"], ["x", "y", "z"])

    inputs = State({"x": jnp.array([1, 2, 3]), "static": "str", "l": 123})

    outs = vmap_fn(inputs)

    expected = {
        "x": jnp.array([1, 2, 3]),
        "y": jnp.array([2, 3, 4]),
        "z": jnp.array([4, 6, 8]),
        "l": 123,
        "static": "str",
    }

    for key in expected:
        if key == "static":
            assert outs[key] == expected[key]
        else:
            assert jnp.allclose(outs[key], expected[key])


def test_grad_works_for_composed_functions():
    @StateFunction.with_output("y")
    def f(w, x):
        return jnp.dot(w, x)

    @StateFunction.with_output("z")
    def g(b, y):
        return b + y

    @StateFunction
    def gof(w, b, x):
        return {"z": w @ x + b, "y": w @ x}

    fn = f | g

    inputs = State(
        {
            "w": jnp.array([1.0, 2.0, 3.0]),
            "x": jnp.array([4.0, 5.0, 6.0]),
            "b": 10.0,
            "static": "str",
        }
    )

    grad_composed = fn.grad(input_keys=["w", "b"], output_key="z")(inputs)
    grad_gof = gof.grad(input_keys=["w", "b"], output_key="z")(inputs)

    grad_expected = {
        "w": jnp.array([1.0, 2.0, 3.0]),
        "x": jnp.array([4.0, 5.0, 6.0]),
        "b": 10.0,
        "static": "str",
        "y": 4.0 + 10.0 + 18.0,
        "z": 4.0 + 10.0 + 18.0 + 10.0,
        "grad_w": jnp.array([4.0, 5.0, 6.0]),
        "grad_b": 1.0,
    }

    assert sorted(grad_composed.keys()) == sorted(grad_expected.keys())
    assert sorted(grad_gof.keys()) == sorted(grad_expected.keys())

    for key in grad_expected:
        if key == "static":
            assert grad_composed[key] == grad_expected[key]
            assert grad_gof[key] == grad_expected[key]
        else:
            assert jnp.allclose(grad_composed[key], grad_expected[key])
            assert jnp.allclose(grad_gof[key], grad_expected[key])
