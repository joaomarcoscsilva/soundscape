from jax import numpy as jnp
import jax
import optax
from functools import partial


from . import sow_transforms


def weighted(loss_fn, class_weights=None):
    """
    Apply weights to a loss function depending on the true labels.
    """

    if class_weights is None:
        return loss_fn

    def _weighted_loss(logits, labels):
        weights = class_weights[labels.argmax(axis=-1)]
        return weights * loss_fn(logits=logits, labels=labels)

    return _weighted_loss


def mean_loss(loss_fn):
    """
    Transform a loss function to return the mean loss.
    """

    def _loss(*args, **kwargs):
        return loss_fn(*args, **kwargs).mean()

    return _loss


def applied_loss(loss_fn, apply_fn):
    """
    Merge a loss function with an apply function.
    """

    def _applied_loss(params, *args, labels, **kwargs):
        logits = apply_fn(params, *args, **kwargs)
        loss = loss_fn(logits=logits, labels=labels)
        return loss

    return _applied_loss


def update(loss_fn, optimizer):
    """
    Create an update function from a loss function and an optimizer.
    """

    def _update(optim_state, params, *args, labels, **kwargs):
        grad = jax.grad(loss_fn)(params, *args, **kwargs, labels=labels)
        updates, optim_state = optimizer.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)
        return optim_state, params

    return _update
