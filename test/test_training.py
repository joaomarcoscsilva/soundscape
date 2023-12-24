import jax
import optax
from jax import numpy as jnp

from soundscape import composition, loss, training


@composition.Composable
def model(values):
    params = values["params"]
    x = values["x"]
    y = x @ params["a"] + params["b"]
    return {**values, "y": y}


def mse(values):
    y = values["y"]
    target = values["target"]
    return jnp.mean((y - target) ** 2)


find_loss = composition.Composable(loss.mean(mse), "loss")

predict = model | find_loss
gradient_fn = composition.grad(predict, "params", "loss")


def test_update():
    optim = optax.sgd(0.1)
    update_fn = gradient_fn | training._get_update_fn(optim)

    params = {"a": jnp.array([2.0]), "b": jnp.array([1.0])}
    optim_state = optim.init(params)

    values = {
        "params": params,
        "x": jnp.array([1.0]),
        "target": jnp.array([5.0]),
        "optim_state": optim_state,
    }

    new_values = update_fn(values)

    assert (
        new_values["params"]["a"]
        == values["params"]["a"] - new_values["grad"]["a"] * 0.1
    )

    loss_value = new_values["loss"]
    new_loss_value = predict(new_values)["loss"]

    assert new_loss_value < loss_value


def test_convergence():
    optim = optax.adam(0.1)

    update_fn = gradient_fn | training._get_update_fn(optim)

    params = {"a": jnp.ones((32, 1)), "b": jnp.zeros((32,))}
    optim_state = optim.init(params)

    seed = jax.random.PRNGKey(0)
    random_x = jax.random.normal(seed, (32, 32))
    random_target = jax.random.normal(seed, (32,))

    values = {
        "params": params,
        "x": random_x,
        "target": random_target,
        "optim_state": optim_state,
    }

    update_fn = composition.jit(update_fn)

    for _ in range(1000):
        values = update_fn(values)

    print(values["loss"])
    assert jnp.allclose(values["loss"], 0.0, atol=1e-3)
