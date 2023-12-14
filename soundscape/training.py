import jax
import optax
from jax import numpy as jnp

from .composition import StateFunction


def update(optimizer: optax.GradientTransformation):
    """
    Return a composable function that updates the parameters and optimizer state.

    The gradient and parameters are expected to be in the values dictionary.

    Parameters:
    ----------
    optimizer: optax.GradientTransformation
        The optimizer to use.
    """

    @StateFunction
    def _update(params: ArrayTree, optim_state: ArrayTree, grad: ArrayTree):
        updates, optim_state = optimizer.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return {"params": params, "optim_state": optim_state}

    return _update
