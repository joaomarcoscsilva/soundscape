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
        w = w.reshape((-1,) + (1,) * (v.ndim - 1))
        return v * w

    def _weighted(batch, outputs: Predictions) -> jax.Array:
        w = weights[batch["labels"]]
        outs = metric_fn(batch, outputs)
        outs = {k: _weight(w, v) for k, v in outs.items()}
        return outs

    return _weighted


def logits(out_key, logits_key="logits") -> MetricFn:
    def _logits(batch: Batch, outputs: Predictions) -> dict:
        return {out_key: outputs[logits_key]}

    return _logits


def preds(out_key, logits_key="logits") -> MetricFn:
    def _preds(batch: Batch, outputs: Predictions) -> dict:
        return {out_key: jnp.argmax(outputs[logits_key], axis=-1)}

    return _preds


def probs(out_key, logits_key="logits") -> MetricFn:
    def _probs(batch: Batch, outputs: Predictions) -> dict:
        return {out_key: jax.nn.softmax(outputs[logits_key])}

    return _probs


def crossentropy(out_key, logits_key="logits") -> MetricFn:
    def _crossentropy(batch: Batch, outputs: Predictions) -> dict:
        ce = optax.softmax_cross_entropy(outputs[logits_key], batch["label_probs"])
        return {out_key: ce}

    return _crossentropy


def brier(out_key, logits_key="logits") -> MetricFn:
    def _brier(batch: Batch, outputs: Predictions) -> dict:
        pred_probs = jax.nn.softmax(outputs[logits_key])
        return {out_key: jnp.sum((pred_probs - batch["label_probs"]) ** 2, axis=-1)}

    return _brier


def accuracy(out_key, logits_key="logits") -> MetricFn:
    def _accuracy(batch: Batch, outputs: Predictions) -> dict:
        return {out_key: outputs[logits_key].argmax(-1) == batch["labels"]}

    return _accuracy


def get_metrics_function(weights):

    return compose(
        [
            weighted(logits("logits_w"), weights),
            preds("preds"),
            preds("preds_w", "logits_w"),
            accuracy("acc"),
            accuracy("acc_w", "logits_w"),
            weighted(accuracy("bal_acc"), weights),
            weighted(accuracy("bal_acc_w"), weights),
            crossentropy("ce_loss"),
            brier("brier"),
        ]
    )
