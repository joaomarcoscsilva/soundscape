from functools import cache

from jax import numpy as jnp

from .. import metrics
from ..types import Batch, ModelState, Predictions
from .base_model import Model


@cache
def _get_model(w_dim, b_dim):
    def _call(batch: Batch, model_state: ModelState, is_training: bool = True):
        logits = batch["inputs"]

        if isinstance(w_dim, int) and w_dim > 0:
            w = model_state.params["w"]
            logits = logits * w

        if b_dim > 0:
            b = model_state.params["b"]
            logits = logits + b

        return Predictions(logits=logits), model_state

    model = Model(_call, metrics.crossentropy("loss"))

    return model


def calibrator(num_epochs, w_dims, b_dim):
    model = _get_model(w_dims, b_dim)

    params = {
        "w": jnp.ones((num_epochs, 1, w_dims)),
        "b": jnp.zeros((num_epochs, 1, b_dim)),
    }
    model_state = ModelState(params, None, None, None)

    return model, model_state
