from threading import local
import jax
from jax import numpy as jnp
import pytest
from oryx.core.interpreters.harvest import call_and_reap

from soundscape.lib import model
from soundscape.lib.settings import settings


def apply_fn(params, x):
    return params["module"]["w"] * x + params["module"]["b"]


def test_partition():

    params = {"module": {"w": jnp.array([1.0]), "b": jnp.array([2.0])}}

    partitioned_apply_fn, part_params, fixed_params = model.partition(
        apply_fn, lambda m, n, p: True, params
    )

    assert partitioned_apply_fn(part_params, fixed_params, 1.0) == apply_fn(params, 1.0)
    assert part_params == params
    assert fixed_params == {}

    partitioned_apply_fn, part_params, fixed_params = model.partition(
        apply_fn, lambda m, n, p: n == "w", params
    )

    assert partitioned_apply_fn(part_params, fixed_params, 1.0) == apply_fn(params, 1.0)
    assert part_params == {"module": {"w": jnp.array([1.0])}}
    assert fixed_params == {"module": {"b": jnp.array([2.0])}}


def test_flax2haiku():
    def flax_apply(variables, x, mutable=False):
        return (
            variables["params"] * variables["batch_stats"] * x,
            {"batch_stats": variables["batch_stats"]},
        )

    variables = {"params": jnp.array([2.0]), "batch_stats": jnp.array([3.0])}
    x = jnp.array([5.0])

    haiku_apply, params, state = model.flax2haiku(flax_apply, variables)

    y_flax, new_state_flax = flax_apply(variables, x)
    y_haiku, new_state_haiku = haiku_apply(params, state, None, x)

    assert y_flax == y_haiku == 30.0
    assert new_state_flax["batch_stats"] == new_state_haiku == state


@pytest.mark.slow
def test_resnet():
    local_settings = settings.copy()
    local_settings["model"]["name"] = "resnet_50"

    (params, fixed_params, state), apply = model.resnet(
        local_settings, jax.random.PRNGKey(12)
    )
    apply = call_and_reap(apply, tag="output")

    y, aux = apply(
        params,
        fixed_params,
        state,
        None,
        jnp.zeros((1, 224, 224, 3)),
        is_training=False,
    )

    assert y.shape == (1, local_settings["data"]["num_classes"])
    assert "state" in aux
    assert aux["state"] == state

    y, aux = apply(
        params,
        fixed_params,
        state,
        None,
        jnp.ones((1, 224, 224, 3)),
        is_training=True,
    )

    assert y.shape == (1, local_settings["data"]["num_classes"])
    assert "state" in aux

    assert jax.tree_util.tree_map(lambda x, y: x.shape == y.shape, aux["state"], state)
    assert jax.tree_util.tree_map(lambda x, y: (x != y).any(), aux["state"], state)
