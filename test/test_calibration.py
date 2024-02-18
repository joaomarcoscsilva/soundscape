import jax
import optax
import pytest
from jax import numpy as jnp

from soundscape import calibrate, metrics
from soundscape.types import Batch


def ce(logits, label_probs):
    return optax.softmax_cross_entropy(logits, label_probs).mean()


def get_new_ce(logits, label_probs, cal_model, cal_state):
    batch = Batch({"inputs": logits, "label_probs": label_probs})
    preds, _ = cal_model(batch, cal_state)
    new_ce = ce(preds["logits"], label_probs)
    return new_ce


@pytest.mark.slow
def test_calibration():

    rng = jax.random.PRNGKey(0)
    rng_logits, rng_labels, rng = jax.random.split(rng, 3)

    logits = jax.random.normal(rng_logits, (100, 2000, 12))
    labels = jax.random.randint(rng_labels, (100, 2000), 0, 12)
    label_probs = jax.nn.one_hot(labels, 12)

    cal_states, cal_models = calibrate.calibrate_all(
        logits, label_probs, lr=2, num_steps=100
    )

    orig_ce = ce(logits, label_probs)

    for cal_name in cal_states:
        cal_state = cal_states[cal_name]
        cal_model = cal_models[cal_name]
        new_ce = get_new_ce(logits, label_probs, cal_model, cal_state)

        if cal_name:
            assert new_ce < orig_ce
        else:
            assert new_ce == orig_ce


@pytest.mark.slow
def test_single_class():

    logits = jax.random.uniform(jax.random.PRNGKey(0), (100, 2000, 12))
    labels = jnp.ones((100, 2000), dtype=jnp.int32)
    label_probs = jax.nn.one_hot(labels, 12)

    cal_states, cal_models = calibrate.calibrate_all(
        logits, label_probs, lr=2, num_steps=100
    )

    orig_ce = ce(logits, label_probs)

    for cal_name in cal_states:
        cal_state = cal_states[cal_name]
        cal_model = cal_models[cal_name]
        new_ce = get_new_ce(logits, label_probs, cal_model, cal_state)

        if "b" in cal_name or "v" in cal_name:
            assert new_ce < 1e-1
        elif cal_name == "t":
            assert new_ce < orig_ce
            assert new_ce > 1
        else:
            assert new_ce == orig_ce


@pytest.mark.slow
def test_uniform():
    logits = jax.nn.one_hot(jnp.zeros((100, 2000), dtype=jnp.int32), 12)
    logits = logits * jax.random.uniform(jax.random.PRNGKey(0), (100, 2000, 1)) * 20

    label_probs = jnp.ones((100, 2000, 12)) / 12

    cal_states, cal_models = calibrate.calibrate_all(
        logits, label_probs, lr=2, num_steps=100
    )

    orig_ce = ce(logits, label_probs)
    best_ce = jnp.log(12)

    for cal_name in cal_states:
        cal_state = cal_states[cal_name]
        cal_model = cal_models[cal_name]
        new_ce = get_new_ce(logits, label_probs, cal_model, cal_state)

        if "v" in cal_name or "t" in cal_name:
            assert jnp.abs(best_ce - new_ce) < 1e-1
        elif cal_name == "b":
            assert new_ce < orig_ce
            assert new_ce > best_ce + 1e-5
        else:
            assert new_ce == orig_ce
