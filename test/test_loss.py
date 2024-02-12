import jax
import numpy as np
from jax import numpy as jnp

from soundscape import loss

jax.config.update("jax_platform_name", "cpu")


def assert_all_close(tree1, tree2):
    (vals1, _) = jax.tree_util.tree_flatten(tree1)
    (vals2, _) = jax.tree_util.tree_flatten(tree2)
    for v1, v2 in zip(vals1, vals2):
        assert jnp.allclose(v1, v2)


def test_logits():
    outputs = {"logits": 123}

    logits = loss.logits("name")({}, outputs)

    assert logits == {"name": 123}


def test_balanced_logits():
    ws = jnp.array([1, 2, 3])
    batch = {"labels": jnp.array([0, 1, 1, 2])}
    outputs = {"logits": jnp.ones((4, 3))}

    bal_logits = loss.weighted(loss.logits("logits_w"), ws)(batch, outputs)

    expected = {"logits": jnp.array([[1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3]])}
    assert_all_close(bal_logits, expected)


def test_crossentropy():
    batch = {"label_probs": jnp.eye(4)}
    outputs = {"logits": (jnp.arange(16).reshape(4, 4) - 8) / 8}

    ce = loss.crossentropy("ce_loss")(batch, outputs)

    probs = jax.nn.softmax(outputs["logits"])
    ce2 = jnp.diag(-jnp.log(probs))
    expected = {"ce_loss": ce2}
    assert_all_close(ce, expected)


def test_brier():
    batch = {"label_probs": jnp.eye(4)}
    outputs = {"logits": (jnp.arange(16).reshape(4, 4) - 8) / 8}

    brier = loss.brier("brier")(batch, outputs)

    probs = jax.nn.softmax(outputs["logits"])
    brier2 = jnp.sum((probs - batch["label_probs"]) ** 2, axis=-1)
    expected = {"brier": brier2}
    assert_all_close(brier, expected)


def test_probs():
    outputs = {
        "logits": jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    }

    probs = loss.probs("probs")({}, outputs)

    expected = {"probs": jax.nn.softmax(outputs["logits"])}
    assert_all_close(probs, expected)


def test_preds():
    outputs = {
        "logits": jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    }

    preds = loss.preds("preds")({}, outputs)

    expected = {"preds": jnp.array([2, 1, 0])}
    assert_all_close(preds, expected)


def test_accuracy():
    batch = {"labels": jnp.array([2, 0, 1, 1, 0])}
    outputs = {
        "logits": jnp.array(
            [
                [0.1, 0.2, 0.7],
                [7.2, 5.2, 2.1],
                [10.1, 0.1, 0.8],
                [0.05, 0.59, 0.1],
                [10, 50, 1],
            ]
        )
    }

    acc = loss.accuracy("acc")(batch, outputs)

    expected = {"acc": jnp.array([1, 1, 0, 1, 0])}
    assert_all_close(acc, expected)


def test_balanced_accuracy():
    batch = {"labels": jnp.array([1, 1, 0])}
    outputs = {
        "logits": jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    }
    class_weights = jnp.array([2, 1, 1])

    bal_acc = loss.weighted(loss.accuracy("bal_acc"), class_weights)(batch, outputs)
    bal_acc_no_weights = loss.weighted(loss.accuracy("bal_acc"), None)(batch, outputs)

    assert jnp.allclose(bal_acc["bal_acc"], jnp.array([0, 1, 2]))
    assert jnp.allclose(bal_acc_no_weights["bal_acc"], jnp.array([0, 1, 1]))


def test_compose():
    batch = {"label_probs": jnp.eye(4)}
    outputs = {"logits": (jnp.arange(16).reshape(4, 4) - 8) / 8}

    ce_fn = loss.crossentropy("ce_loss")
    brier_fn = loss.brier("brier")
    composed = loss.compose([ce_fn, brier_fn])(batch, outputs)

    ce = loss.crossentropy("ce_loss")(batch, outputs)["ce_loss"]
    brier = loss.brier("brier")(batch, outputs)["brier"]
    expected = {"ce_loss": ce, "brier": brier, "logits": outputs["logits"]}
    assert_all_close(composed, expected)
