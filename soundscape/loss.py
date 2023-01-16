from typing import Optional
from jax import numpy as jnp
import jax
from .composition import SimpleFunction, Composable
import optax


def kl(p1, p2):
    return p1 * jnp.log(p1 / p2)


def js(ps):
    ps = jnp.clip(ps, 1e-7, 1)
    pm = ps.mean(axis=0)
    n = ps.shape[0]
    return kl(ps, pm).mean(axis=1)


def crossentropy(values):
    return optax.softmax_cross_entropy(logits=values["logits"], labels=values["labels"])


def preds(values):
    return values["logits"].argmax(axis=-1)


def augmix_loss(loss_fn: SimpleFunction, num_repetitions=3, l=1.0) -> Composable:
    @Composable
    def augmix_loss(values):
        logits = values["logits"]
        labels = values["labels"]

        logits = jnp.split(logits, num_repetitions)

        ce_loss = loss_fn(
            {**values, "logits": logits[0], "labels": labels[: logits.shape[1]]}
        )

        probs = jax.nn.softmax(logits)
        js_loss = js(probs)

        loss = ce_loss + l * js_loss

        return {**values, "loss": loss, "js_loss": js_loss, "ce_loss": ce_loss}

    return augmix_loss


def accuracy(values):
    return (values["preds"] == values["labels"].argmax(axis=-1)).mean()


def weighted(metric_function: SimpleFunction, class_weights=None) -> SimpleFunction:

    if class_weights is None:
        return metric_function

    return (
        lambda values: metric_function(values)
        * class_weights[values["labels"].argmax(axis=-1)]
    )


def mean(function: SimpleFunction) -> SimpleFunction:
    return lambda values: function(values).mean()
