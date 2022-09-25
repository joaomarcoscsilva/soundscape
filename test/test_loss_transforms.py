import pytest
import optax
import jax
from jax import numpy as jnp
from oryx.core.interpreters.harvest import reap, call_and_reap
from soundscape.lib import loss_transforms, sow_transforms
from functools import partial


@partial(sow_transforms.sow_fn, name="loss", tag="tag")
def loss_fn(logits, labels):
    return logits


@partial(sow_transforms.tuple_to_sow, names=["logits", "state"], tag="tag")
def logit_fn(params, x):
    return (params["w"] - x) ** 2, {}


@pytest.mark.parametrize(
    "logits,labels,class_weights,expected",
    [
        (
            jnp.array([2, 2, 2]),
            jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            jnp.array([1, 2, 3]),
            jnp.array([2, 6, 4]),
        ),
        (
            jnp.array([2, 2, 2]),
            jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            None,
            jnp.array([2, 2, 2]),
        ),
    ],
)
def test_weighted(logits, labels, class_weights, expected):
    weighted_loss = loss_transforms.weighted(loss_fn, class_weights)
    assert jnp.allclose(weighted_loss(logits, labels), expected)


@pytest.mark.parametrize(
    "params,inputs,expected_loss,expected_logits,expected_grad",
    [
        (
            {"w": jnp.array([4.0])},
            jnp.array([1.0]),
            jnp.array([9.0]),
            jnp.array([9.0]),
            jnp.array([6.0]),
        )
    ],
)
def test_applied_loss_and_update(
    params, inputs, expected_loss, expected_logits, expected_grad
):

    # Test applied_loss
    applied_loss = loss_transforms.applied_loss(loss_fn, logit_fn)

    loss = applied_loss(params, inputs, labels=None)
    assert jnp.allclose(loss, expected_loss)

    # Test planted logits
    aux = reap(applied_loss, tag="tag")(params, inputs, labels=None)
    assert jnp.allclose(aux["logits"], expected_logits)

    # Test update_from_loss
    optim = optax.sgd(0.1)
    optim_state = optim.init(params)

    update_fn = loss_transforms.update(applied_loss, optim)

    (optim_state, new_params), aux = call_and_reap(update_fn, tag="tag")(
        optim_state, params, inputs, labels=None
    )

    assert jnp.allclose(aux["loss"], expected_loss)
    assert jnp.allclose(aux["logits"], expected_logits)
    assert jnp.allclose(new_params["w"], params["w"] - 0.1 * expected_grad)

    # Check that the loss decreased after the update
    new_loss = applied_loss(new_params, inputs, labels=None)

    assert new_loss < expected_loss
