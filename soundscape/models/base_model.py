from typing import Any, Callable, Literal, NamedTuple, Optional, TypedDict, Union

import jax

from ..dataset.dataloading import Batch

PyTree = Any

partition_fns = {
    "all": lambda m, n, p: True,
    "none": lambda m, n, p: False,
    "head": lambda m, n, p: "logits" in m,
}


class ModelState(NamedTuple):
    params: PyTree
    fixed_params: PyTree

    state: Optional[PyTree]
    optim_state: Optional[PyTree]


class Predictions(TypedDict, total=False):
    logits: PyTree
    ...


def initialize_optim_state(model_state: ModelState, optimizer: Any) -> ModelState:
    return model_state._replace(optim_state=optimizer.init(model_state.params))


class Model:
    """
    Wrapper class that contains the model state and functions related to it.
    """

    def __init__(
        self,
        predict_fn: Callable[[Batch, ModelState, bool], (Predictions, ModelState)],
        loss_fn: Callable[[Batch, Predictions], float],
    ):

        self.__call__ = predict_fn

        def _predict_with_loss(batch, params, model_state, is_training):
            model_state = model_state._replace(params=params)
            outputs, model_state = self(batch, model_state, is_training)
            loss = loss_fn(batch, outputs)
            return loss, (outputs, model_state)

        self._grad_fn = jax.value_and_grad(_predict_with_loss, has_aux=True, argnums=1)

    def value_and_grad(
        self,
        batch: Batch,
        model_state: ModelState,
        is_training: bool = True,
    ) -> tuple[Predictions, ModelState, PyTree]:

        (loss, (outputs, model_state)), grads = self._grad_fn(
            batch, model_state.params, model_state, is_training
        )

        return outputs, model_state, grads


model_creators = {}
