from typing import Callable

import jax
import optax
from jax import numpy as jnp

from .types import Batch, Predictions

MetricFn = Callable[[Batch, Predictions], dict]


def compose(fns: list[MetricFn]):
    def _compose(batch: Batch, outputs: Predictions) -> jax.Array:
        for fn in fns:
            outputs = outputs | fn(batch, outputs)

        return outputs

    return _compose


def weighted(metric_fn: MetricFn, weights=jnp.ones(1)) -> MetricFn:
    if weights is None:
        return metric_fn

    def _weight(w, v):
        return v * w

    def _weighted(batch, outputs: Predictions) -> jax.Array:
        w = weights[batch["labels"]]
        outs = metric_fn(batch, outputs)
        outs = {k: _weight(w, v) for k, v in outs.items()}
        return outs

    return _weighted


def probs(out_key, weights=1) -> MetricFn:
    def _probs(batch: Batch, outputs: Predictions) -> dict:
        return {out_key: jax.nn.softmax(outputs["logits"]) * weights}

    return _probs


def accuracy(out_key, probs_key="probs") -> MetricFn:
    def _accuracy(batch: Batch, outputs: Predictions) -> dict:
        return {out_key: outputs[probs_key].argmax(-1) == batch["labels"]}

    return _accuracy


def crossentropy(out_key) -> MetricFn:
    def _crossentropy(batch: Batch, outputs: Predictions) -> dict:
        ce = optax.softmax_cross_entropy(outputs["logits"], batch["label_probs"])
        return {out_key: ce}

    return _crossentropy


def brier(out_key) -> MetricFn:
    def _brier(batch: Batch, outputs: Predictions) -> dict:
        pred_probs = jax.nn.softmax(outputs["logits"])
        return {out_key: jnp.sum((pred_probs - batch["label_probs"]) ** 2, axis=-1)}

    return _brier


def get_metrics_function(weights):

    return compose(
        [
            probs("probs"),
            probs("probs_w", weights),
            accuracy("acc"),
            accuracy("acc_w", "probs_w"),
            weighted(accuracy("bal_acc"), weights),
            weighted(accuracy("bal_acc_w", "probs_w"), weights),
            crossentropy("ce_loss"),
            brier("brier"),
        ]
    )
