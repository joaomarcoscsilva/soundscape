from soundscape import loss

import jax
from jax import numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")


def test_kl():
    p1 = jnp.array([0.25, 0.25, 0.25, 0.25])
    p2 = jnp.array([0.5, 0.25, 0.125, 0.125])

    assert jnp.allclose(loss.kl(p1, p1), jnp.zeros(4))
    assert jnp.allclose(loss.kl(p2, p2), jnp.zeros(4))

    kl1 = loss.kl(p1, p2) / jnp.log(2)
    kl2 = loss.kl(p2, p1) / jnp.log(2)

    assert jnp.allclose(kl1, jnp.array([-0.25, 0, 0.25, 0.25]))
    assert jnp.allclose(kl2, jnp.array([0.5, 0, -0.125, -0.125]))


def test_js():
    p1 = jnp.array([0.25, 0.25, 0.25, 0.25])
    p2 = jnp.array([0.5, 0.25, 0.125, 0.125])
    pm = (p1 + p2) / 2

    kl1 = loss.kl(p1, pm)
    kl2 = loss.kl(p2, pm)
    js1 = loss.js(jnp.stack([p1, p2]))
    js2 = loss.js(jnp.stack([p2, p1]))

    assert jnp.allclose(js1, (kl1 + kl2) / 2)
    assert jnp.allclose(js2, js1)


def test_crossentropy():
    logits = (jnp.arange(16).reshape(4, 4) - 8) / 8
    labels = jnp.eye(4)

    ce = loss.crossentropy({"logits": logits, "one_hot_labels": labels})

    probs = jax.nn.softmax(logits)
    ce2 = jnp.diag(-jnp.log(probs))

    assert jnp.allclose(ce, ce2)


def test_preds():
    logits = jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    preds = loss.preds()({"logits": logits})
    assert jnp.allclose(preds, jnp.array([2, 1, 0]))


def test_weighted_preds():
    logits = jnp.ones((10, 3))
    weights = jnp.array([1.0, 4.0, 3.0])
    preds = loss.preds(weights)({"logits": logits})
    assert jnp.allclose(preds, 1)


def test_accuracy():
    logits = jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    preds = loss.preds()({"logits": logits})
    labels = jnp.array([1, 1, 0])
    acc = loss.accuracy({"preds": preds, "labels": labels})
    assert jnp.allclose(acc, jnp.array([0, 1, 1]))


def test_weighted():
    logits = jnp.array([[1.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    preds = loss.preds()({"logits": logits})
    labels = jnp.array([2, 1, 0])

    weights = jnp.array([1.0, 2.0, 3.0])

    weighted_acc = loss.weighted(loss.accuracy, weights)

    acc = weighted_acc({"preds": preds, "labels": labels})

    assert jnp.allclose(acc, jnp.array([0.0, 2.0, 1.0]))

    assert loss.weighted(loss.accuracy, None) == loss.accuracy


def test_mean():
    logits = jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    preds = loss.preds()({"logits": logits})
    labels = jnp.array([1, 1, 0])

    acc_fn = loss.mean(loss.accuracy)

    acc = acc_fn({"preds": preds, "labels": labels})

    assert jnp.allclose(acc, 2 / 3)
