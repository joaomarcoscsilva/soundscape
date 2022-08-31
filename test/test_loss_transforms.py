import pytest
import optax
import jax
import equinox as eqx
from jax import numpy as jnp

import loss_transforms


class model(eqx.Module):
    w: jnp.array

    def __init__(self, dim):
        self.w = jnp.eye(dim)

    def __call__(self, x, key):
        return self.w @ x


def l(logits, labels):
    return logits


@pytest.mark.parametrize(
    "logits,labels,class_weights,expected",
    [
        (
            jnp.array([2, 2, 2]),
            jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            jnp.array([1, 2, 3]),
            jnp.array([2, 6, 4]),
        ),
    ],
)
def test_weighted_loss(logits, labels, class_weights, expected):
    weighted_loss = loss_transforms.weighted_loss(l, class_weights)
    assert jnp.allclose(weighted_loss(logits, labels), expected)


@pytest.mark.parametrize(
    "model,inputs,expected_loss,expected_logits,expected_grad",
    [
        (
            model(dim=3),
            jnp.array([[1, 2, 3]]),
            jnp.array(2),
            jnp.array([[1, 2, 3]]),
            jnp.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) * 1 / 3,
        )
    ],
)
def test_applied_loss_and_update(
    model, inputs, expected_loss, expected_logits, expected_grad
):
    key = jax.random.PRNGKey(0)[None, :]
    labels = 0

    # Test applied_loss
    applied_loss = loss_transforms.applied_loss(l)
    (loss, logits) = applied_loss(model, inputs, labels, key)

    assert jnp.allclose(loss, expected_loss)
    assert jnp.allclose(logits, expected_logits)

    # Test the gradient of applied_loss
    grad_loss = eqx.filter_value_and_grad(applied_loss, has_aux=True)
    (loss, logits), grad = grad_loss(model, inputs, labels, key)

    assert jnp.allclose(loss, expected_loss)
    assert jnp.allclose(logits, expected_logits)
    assert jnp.allclose(grad.w, expected_grad)

    # Test update_from_loss
    update_fn = loss_transforms.update_from_loss(grad_loss)

    optim = optax.sgd(1)
    optim_state = optim.init(model)

    new_model, optim_state, loss, logits = update_fn(
        model, inputs, labels, optim, optim_state, key
    )

    assert jnp.allclose(loss, expected_loss)
    assert jnp.allclose(logits, expected_logits)
    assert jnp.allclose(new_model.w, model.w - expected_grad)

    # Check that the loss decreased after the update
    (new_loss, new_logits) = applied_loss(new_model, inputs, labels, key)

    assert new_loss < expected_loss
