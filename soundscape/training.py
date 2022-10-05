from jax import numpy as jnp
import jax
import optax
from .composition import Composable


def update(optimizer: optax.GradientTransformation):
    """
    Return a composable function that updates the parameters and optimizer state.

    The gradient and parameters are expected to be in the values dictionary.

    Parameters:
    ----------
    optimizer: optax.GradientTransformation
        The optimizer to use.
    """

    @Composable
    def _update(values):
        """
        Update the parameters and optimizer state.

        Parameters:
        ----------
        values["params"]: jnp.ndarray
            The parameters to update.
        values["optim_state"]: optax.OptState
            The optimizer state.
        values["grad"]: jnp.ndarray
            The gradient to use for the update.

        """
        params = values["params"]
        optim_state = values["optim_state"]
        grad = values["grad"]

        updates, optim_state = optimizer.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return {**values, "params": params, "optim_state": optim_state}

    return _update
