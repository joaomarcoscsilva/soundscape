import jax
import optax
from jax import numpy as jnp

from .composition import StateFunction
from .typechecking import Array


@StateFunction.with_output("ce")
def crossentropy(logits: Array, label_probs: Array) -> Array:
    return optax.softmax_cross_entropy(logits=logits, labels=label_probs)


@StateFunction.with_output("brier")
def brier(logits: Array, label_probs: Array) -> Array:
    pred_probs = jax.nn.softmax(logits)
    return jnp.sum((pred_probs - label_probs) ** 2, axis=-1)


@StateFunction.with_output("probs")
def probs(logits: Array) -> Array:
    return jax.nn.softmax(logits)


@StateFunction.with_output("preds")
def preds(probs: Array) -> Array:
    return probs.argmax(axis=-1)


@StateFunction.with_output("acc")
def accuracy(preds: Array, labels: Array) -> Array:
    return preds == labels


@StateFunction.with_output("metric")
def weight_metric(
    metric: Array, labels: Array, class_weights: None | Array = jnp.ones(1)
) -> Array:
    if class_weights is None:
        return metric

    return metric * class_weights[labels]


@StateFunction.with_output("metric")
def mean_metric(metric: Array) -> Array:
    return metric.mean()
