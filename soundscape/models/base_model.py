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

        self.__call__ = predict_fn

        def _predict_with_loss(batch, params, model_state, is_training):
            model_state = model_state._replace(params=params)
            outputs, model_state = self(batch, model_state, is_training)
            loss = loss_fn(batch, outputs)["loss"]
            return loss, (outputs, model_state)

        self._grad_fn = jax.value_and_grad(_predict_with_loss, has_aux=True, argnums=1)

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
