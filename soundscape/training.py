import jax
import optax
from jax import numpy as jnp

from soundscape import log


def _get_update_fn(optimizer: optax.GradientTransformation):
    """
    Return a function that updates the parameters and optimizer state.
    """

    def _update_fn(state, grad):
        params = state["params"]
        optim_state = state["optim_state"]

        updates, optim_state = optimizer.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return state | {"params": params, "optim_state": optim_state}

    return _update_fn


def scan_dataset(scan_fn, state, dataset, aggregate_mode="concat"):
    preds_list = []
    for batch in dataset:
        state, preds = scan_fn(state, batch)
        preds_list.append(preds)
    preds_list = log.merge(preds_list, aggregate_mode)
    return state, preds_list


def call_fn(preprocess_batch, call_fn, postprocess_preds=lambda x: x):
    def _call_fn(state, batch):
        batch = preprocess_batch(batch)
        preds = call_fn(state, batch)
        preds = postprocess_preds(preds)
        return state, preds

    return _call_fn


def train_fn(preprocess_batch, call_grad_fn, optimizer, postprocess_preds=lambda x: x):
    update_fn = _get_update_fn(optimizer)

    def _train_fn(state, preds):
        batch = preprocess_batch(batch)
        preds, grads = call_grad_fn(state, batch)
        state = update_fn(state, grads)
        preds = postprocess_preds(preds)
        return state, preds

    return _train_fn


def train_epoch(train_fn, call_fn, dataset, batch_size, eval_test=False):
    def _train_for_epoch(rng):
        state, epoch_train_results = scan_dataset(
            train_fn, state, dataset.iterate(rng, "train", batch_size)
        )

        _, epoch_eval_results = scan_dataset(
            call_fn, state, dataset.iterate(rng, "val", batch_size)
        )

        epoch_results = {"train": epoch_train_results, "val": epoch_eval_results}

        if eval_test:
            _, epoch_test_results = scan_dataset(
                call_fn, state, dataset.iterate(rng, "test", batch_size)
            )
            epoch_results["test"] = epoch_test_results

        return state, epoch_results


def train(settings):
    def _train(dataset, model):

        metrics = ...
        model = ...

        return model, metrics

    return _train
