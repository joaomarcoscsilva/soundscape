from typing import Callable

import jax

from .. import metrics
from ..types import Batch, ModelState, Predictions, PyTree

partition_fns = {
    "all": lambda m, n, p: True,
    "none": lambda m, n, p: False,
    "head": lambda m, n, p: "logits" in m,
}


class Model:
    """
    Wrapper class that contains the model state and functions related to it.
    """

    def __init__(
        self,
        predict_fn: Callable[[Batch, ModelState, bool], tuple[Predictions, ModelState]],
        loss_fn: Callable[[Batch, Predictions], dict],
    ):

        self.call_fn = jax.jit(predict_fn, static_argnums=2)

        def _predict_with_loss(batch, params, model_state, is_training):
            model_state = model_state._replace(params=params)
            outputs, model_state = self(batch, model_state, is_training)
            loss = loss_fn(batch, outputs)["loss"].mean()
            return loss, (outputs, model_state)

        grad_fn = jax.value_and_grad(_predict_with_loss, has_aux=True, argnums=1)
        self._grad_fn = jax.jit(grad_fn, static_argnums=(3,), donate_argnums=(1, 2))

    def __call__(self, batch, model_state, training):
        if not isinstance(batch, dict):
            batch = {"inputs": batch}

        if not isinstance(model_state, ModelState):
            model_state = ModelState(params=model_state)

        return self.call_fn(batch, model_state, training)

    def value_and_grad(
        self,
        batch: Batch,
        model_state: ModelState,
        is_training: bool = False,
    ) -> tuple[Predictions, ModelState, PyTree]:

        (loss, (outputs, model_state)), grads = self._grad_fn(
            batch, model_state.params, model_state, is_training
        )

        return outputs, model_state, grads


model_creators = {}


def get_model(
    rng,
    num_classes,
    model_settings,
    loss_fn=metrics.crossentropy("loss"),
):
    model_creating_fn = model_creators[model_settings.model_name]
    model, model_state = model_creating_fn(rng, loss_fn, num_classes, model_settings)

    return model, model_state
