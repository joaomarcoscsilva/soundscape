import jax
import numpy as np
import optax
from jax import numpy as jnp

from soundscape import metrics

jax.config.update("jax_platform_name", "cpu")


def assert_all_close(tree1, tree2):
    (vals1, treedef1) = jax.tree_util.tree_flatten(tree1)
    (vals2, treedef2) = jax.tree_util.tree_flatten(tree2)

    assert treedef1 == treedef2
    for v1, v2 in zip(vals1, vals2):
        assert jnp.allclose(v1, v2)


def test_crossentropy():
    batch = {"label_probs": jnp.eye(4)}
    outputs = {"logits": (jnp.arange(16).reshape(4, 4) - 8) / 8}

    ce = metrics.crossentropy("ce_loss")(batch, outputs)

    probs = jax.nn.softmax(outputs["logits"])
    ce2 = jnp.diag(-jnp.log(probs))
    expected = {"ce_loss": ce2}
    assert_all_close(ce, expected)


def test_brier():
    batch = {"label_probs": jnp.eye(4)}
    outputs = {"logits": (jnp.arange(16).reshape(4, 4) - 8) / 8}

    brier = metrics.brier("brier")(batch, outputs)

    probs = jax.nn.softmax(outputs["logits"])
    brier2 = jnp.sum((probs - batch["label_probs"]) ** 2, axis=-1)
    expected = {"brier": brier2}
    assert_all_close(brier, expected)


def test_probs():
    outputs = {
        "logits": jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
    }

    probs = metrics.probs("probs")({}, outputs)

    expected = {"probs": jax.nn.softmax(outputs["logits"])}
    assert_all_close(probs, expected)


def test_accuracy():
    batch = {"labels": jnp.array([2, 0, 1, 1, 0])}
    outputs = {
        "probabilities": jax.nn.softmax(
            jnp.array(
                [
                    [0.1, 0.2, 0.7],
                    [7.2, 5.2, 2.1],
                    [10.1, 0.1, 0.8],
                    [0.05, 0.59, 0.1],
                    [10, 50, 1],
                ]
            )
        )
    }

    acc = metrics.accuracy("acc", "probabilities")(batch, outputs)

    expected = {"acc": jnp.array([1, 1, 0, 1, 0])}
    assert_all_close(acc, expected)


def test_balanced_accuracy():
    batch = {"labels": jnp.array([1, 1, 0])}
    outputs = {
        "probs": jax.nn.softmax(
            jnp.array([[0.1, 0.2, 0.7], [5.2, 7.5, 2.1], [10.1, 0.1, 0.8]])
        )
    }
    class_weights = jnp.array([2, 1, 1])

    bal_acc = metrics.weighted(metrics.accuracy("bal_acc"), class_weights)(
        batch, outputs
    )
    bal_acc_no_weights = metrics.weighted(metrics.accuracy("bal_acc"), None)(
        batch, outputs
    )

    assert jnp.allclose(bal_acc["bal_acc"], jnp.array([0, 1, 2]))
    assert jnp.allclose(bal_acc_no_weights["bal_acc"], jnp.array([0, 1, 1]))


def test_compose():
    batch = {"label_probs": jnp.eye(4)}
    outputs = {"logits": (jnp.arange(16).reshape(4, 4) - 8) / 8}

    ce_fn = metrics.crossentropy("ce_loss")
    brier_fn = metrics.brier("brier")
    composed = metrics.compose([ce_fn, brier_fn])(batch, outputs)

    ce = metrics.crossentropy("ce_loss")(batch, outputs)["ce_loss"]
    brier = metrics.brier("brier")(batch, outputs)["brier"]
    expected = {"ce_loss": ce, "brier": brier, "logits": outputs["logits"]}
    assert_all_close(composed, expected)


def test_metrics_fn():
    batch = {
        "label_probs": jnp.array(
            [
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.5 + 1e-3, 0.5 - 1e-3],
                    [1 / 3 + 2e-3, 1 / 3 - 1e-3, 1 / 3 - 1e-3],
                    [0.9, 0.05, 0.05],
                ]
            ]
        ),
        "labels": jnp.array([[2, 1, 0, 0]]),
    }

    out_probs = jnp.array(
        [
            [0.1, 0.5, 0.4],
            [1 / 3 + 2e-3, 1 / 3 - 1e-3, 1 / 3 - 1e-3],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
        ]
    )
    preds = {"logits": jnp.log(out_probs)}
    class_weights = jnp.array([4, 2, 3])
    ws = jnp.array([3, 2, 4, 4])

    metrics_fn = metrics.get_metrics_function(class_weights)
    outputs = metrics_fn(batch, preds)

    expected = {
        "logits": preds["logits"],
        "probs": out_probs,
        "probs_w": out_probs * class_weights,
        "acc": jnp.array([[0.0, 0.0, 1.0, 1.0]]),
        "acc_w": jnp.array([[1.0, 0.0, 1.0, 1.0]]),
        "bal_acc": jnp.array([[0.0, 0.0, 4.0, 4.0]]),
        "bal_acc_w": jnp.array([[3.0, 0.0, 4.0, 4.0]]),
        "ce_loss": optax.softmax_cross_entropy(
            jnp.log(out_probs), batch["label_probs"]
        ),
        "brier": jnp.sum((out_probs - batch["label_probs"]) ** 2, -1),
    }

    assert_all_close(outputs, expected)
