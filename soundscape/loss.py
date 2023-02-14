from typing import Optional
from jax import numpy as jnp
import jax
from .composition import Composable
import optax


def kl(p1, p2):
    """
    Compute the Kullback-Leibler divergence between two distributions.
    """
    return p1 * jnp.log(p1 / p2)


def js(ps):
    """
    Compute the Jensen-Shannon divergence between a set of distributions.
    """
    ps = jnp.clip(ps, 1e-7, 1)
    pm = ps.mean(axis=0)
    return kl(ps, pm).mean(axis=0)


def crossentropy(values):
    """
    Compute the cross-entropy loss.
    """

    return optax.softmax_cross_entropy(
        logits=values["logits"], labels=values["one_hot_labels"]
    )


def preds(values):
    """
    Compute the predictions from the logits
    """

    return values["logits"].argmax(axis=-1)


def augmix_loss(loss_fn, num_repetitions=3, l=1.0):
    """
    UNTESTED AugMix loss function.
    """

    @Composable
    def augmix_loss(values):
        logits = values["logits"]
        labels = values["one_hot_labels"]

        logits = jnp.split(logits, num_repetitions)

        ce_loss = loss_fn(
            {**values, "logits": logits[0], "one_hot_labels": labels[: logits.shape[1]]}
        )

        probs = jax.nn.softmax(logits)
        js_loss = js(probs)

        loss = ce_loss + l * js_loss

        return {**values, "loss": loss, "js_loss": js_loss, "ce_loss": ce_loss}

    return augmix_loss


def accuracy(values):
    """
    Compute the accuracy from labels and predictions.
    """

    return jnp.float32(values["preds"] == values["labels"])


def weighted(metric_function, class_weights=None):
    """
    Transform a metric function into a weighted metric function, with
    weights depending on the class of each sample.
    """

    if class_weights is None:
        return metric_function

    return lambda values: metric_function(values) * class_weights[values["labels"]]


def mean(function):
    """
    Transform a metric function into a function that computes the mean
    of the metric over a batch.
    """
    return lambda values: function(values).mean()
