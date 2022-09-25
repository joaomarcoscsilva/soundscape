from jax import numpy as jnp
import jax
import optax
from functools import partial
from oryx.core.interpreters.harvest import call_and_reap

from soundscape.lib import supervised, sow_transforms
from soundscape.lib.settings import settings


@partial(sow_transforms.tuple_to_sow, names=["logits", "state"], tag="output")
def apply_fn(params, fixed_params, state, rng, x, *, is_training=False):
    return params["w"] * x + params["b"], state


def test_optim():
    params = {"w": jnp.array([1.0]), "b": jnp.array([2.0])}

    optim, optim_state = supervised.get_optimizer(params)

    loss, grad = jax.value_and_grad(lambda *args: apply_fn(*args).mean())(
        params, None, None, None, 1.0
    )

    updates, optim_state = optim.update(grad, optim_state, params)
    new_params = optax.apply_updates(updates, params)

    assert apply_fn(new_params, None, None, None, 1.0)[0].mean() < loss


def test_get_loss():
    local_settings = settings.copy()
    local_settings["model"]["balanced"] = False

    loss_fn = supervised.get_loss(apply_fn)

    params = {"w": jnp.array([1.0]), "b": jnp.array([2.0])}
    x = jnp.array([1.0])
    y = jnp.array([2.0])

    pred = apply_fn(params, {}, 0.0, None, x)
    l = optax.softmax_cross_entropy(pred, y)

    loss, aux = call_and_reap(loss_fn, tag="output")(params, {}, 0.0, None, x, labels=y)

    assert loss == l
    assert aux["logits"] == pred
    assert "state" in aux

    loss, aux = call_and_reap(loss_fn, tag="model")(params, {}, 0.0, None, x, labels=y)
    assert loss == l
    assert "accuracy" in aux
    assert "balanced_accuracy" in aux
    assert "loss" in aux
    assert "balanced_loss" in aux


def test_get_update():
    params = {"w": jnp.array([1.0]), "b": jnp.array([2.0])}
    x = jnp.array([1.0])
    y = jnp.array([2.0])

    optim, optim_state = supervised.get_optimizer(params)
    loss_fn = supervised.get_loss(apply_fn)
    update_fn = supervised.get_update(loss_fn, optim)

    (new_optim_state, new_params, state), aux = call_and_reap(update_fn, tag="model")(
        optim_state, params, {}, 0.0, None, x, labels=y
    )

    assert new_params != params
    assert "accuracy" in aux
    assert "balanced_accuracy" in aux
    assert "loss" in aux
    assert "balanced_loss" in aux
