import jax
from jax import numpy as jnp

from ..types import Batch, ModelState, Predictions
from .base_model import Model, model_creators


def linear(rng, loss_fn, num_classes, model_settings):
    def call_linear(batch: Batch, model_state: ModelState, is_training: bool = True):
        inputs = batch["inputs"].reshape((batch["inputs"].shape[0], -1))
        logits = jnp.dot(inputs, model_state.params["w"]) + model_state.params["b"]
        return Predictions(logits=logits), model_state

    model = Model(call_linear, loss_fn)

    params = {
        "w": jax.random.normal(rng, (model_settings.input_dim, num_classes)),
        "b": jnp.zeros(num_classes),
    }

    model_state = ModelState(params, None, None, None)

    return model, model_state


model_creators["linear"] = linear
