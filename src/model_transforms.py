from functools import partial, wraps
import equinox as eqx
import jax
from jax import numpy as jnp
import eqxvision
import optax
from typing import Callable


def replace_classifier(
    model: eqx.Module, num_classes: int, rng: jax.random.PRNGKey
) -> eqx.Module:
    """
    Replace the fully connected classifier with a new one.
    """

    return eqx.tree_at(
        where=lambda model_pytree: model_pytree.fc,
        pytree=model,
        replace_fn=lambda fc: eqx.nn.Linear(fc.in_features, num_classes, key=rng),
    )
