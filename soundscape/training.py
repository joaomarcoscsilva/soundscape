from jax import numpy as jnp
import jax
import optax
from .composition import ComposableFunction, Composable


def update(optimizer: optax.GradientTransformation) -> ComposableFunction:
    @Composable
    def _update(values):
        params = values["params"]
        optim_state = values["optim_state"]
        grad = values["grad"]

        updates, optim_state = optimizer.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return {**values, "params": params, "optim_state": optim_state}

    return _update
