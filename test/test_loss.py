import jax
import numpy as np
from jax import numpy as jnp

from soundscape import loss
from soundscape.composition import State

jax.config.update("jax_platform_name", "cpu")


def test_crossentropy():
    logits = (jnp.arange(16).reshape(4, 4) - 8) / 8
    labels = jnp.eye(4)

    ce = loss.crossentropy(State({"logits": logits, "label_probs": labels}))["ce"]

    probs = jax.nn.softmax(logits)
    ce2 = jnp.diag(-jnp.log(probs))

    assert jnp.allclose(ce, ce2)


def test_brier():
    logits = (jnp.arange(16).reshape(4, 4) - 8) / 8
    labels = jnp.eye(4)

    brier = loss.brier(State({"logits": logits, "label_probs": labels}))["brier"]

    probs = jax.nn.softmax(logits)
    brier2 = jnp.sum((probs - labels) ** 2, axis=-1)

    assert jnp.allclose(brier, brier2)


def test_probs():
    logits = jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    probs = loss.probs(State({"logits": logits}))["probs"]
    assert jnp.allclose(probs, jax.nn.softmax(logits))


def test_preds():
    probs = jnp.array([[0.1, 0.2, 0.7], [1.0, 0.0, 0.0], [0.4, 0.3, 0.3]])
    preds = loss.preds(State({"probs": probs}))["preds"]
    assert jnp.allclose(preds, jnp.array([2, 0, 0]))


def test_accuracy():
    preds = jnp.array([2, 0, 0, 1, 1])
    labels = jnp.array([2, 0, 1, 1, 0])
    acc = loss.accuracy(State({"preds": preds, "labels": labels}))["acc"]
    assert jnp.allclose(acc, jnp.array([1, 1, 0, 1, 0]))


def test_weight_metric():
    weight_metric = loss.weight_metric.output("bal_acc").input("acc", "metric")
    fn = loss.probs | loss.preds | loss.accuracy | weight_metric

    logits = jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    labels = jnp.array([1, 1, 0])
    class_weights = jnp.array([2, 1, 1])

    bal_acc = fn(
        State({"logits": logits, "labels": labels, "class_weights": class_weights})
    )["bal_acc"]

    assert jnp.allclose(bal_acc, jnp.array([0, 1, 2]))

    bal_acc_no_weights = fn(
        State({"logits": logits, "labels": labels, "class_weights": None})
    )["bal_acc"]

    assert jnp.allclose(bal_acc_no_weights, jnp.array([0, 1, 1]))


def test_mean():
    preds = jnp.array([2, 0, 0, 1, 1])
    labels = jnp.array([2, 0, 1, 1, 0])

    fn = loss.accuracy | loss.mean_metric.output("mean_acc").input("acc", "metric")

    acc = fn(State({"preds": preds, "labels": labels}))["mean_acc"]

    assert jnp.allclose(acc, 0.6)
