from jax import numpy as jnp
import jax
import optax
from functools import partial


from . import sow_transforms


def weighted(loss_fn, class_weights=None):
    """
    Takes in a loss function and returns a new function that
    weights samples according to their target labels. The weights for each
    class must be passed as an argument to the constructor.
    """

    if class_weights is None:
        return loss_fn

    def _weighted_loss(logits, labels):
        weights = class_weights[labels.argmax(axis=-1)]
        return weights * loss_fn(logits=logits, labels=labels)

    return _weighted_loss


def mean_loss(loss_fn):
    def _loss(*args, **kwargs):
        return loss_fn(*args, **kwargs).mean()

    return _loss


def applied_loss(loss_fn, logits_fn):
    """
    Takes in a loss function and a function that produces logits
    and returns a function that combine the two. The loss function
    and any auxiliary    metrics are reduced before being returned.
    """

    def _applied_loss(params, *args, labels, **kwargs):
        logits = logits_fn(params, *args, **kwargs)
        loss = loss_fn(logits=logits, labels=labels)
        return loss

    return _applied_loss


def update(loss_fn, optimizer):
    """
    Takes in a loss function transformed by jax.value_and_grad
    and an optimizer and returns an update function.
    """

    def _update(optim_state, params, *args, labels, **kwargs):
        grad = jax.grad(loss_fn)(params, *args, **kwargs, labels=labels)
        updates, optim_state = optimizer.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)
        return optim_state, params

    return _update
