import jax
import pytest
from jax import numpy as jnp

from soundscape import calibrate, metrics
from soundscape.types import Batch

cal_config = {"lr": 2, "num_steps": 100}

metrics_fn = metrics.get_metrics_function(jnp.arange(12))


@pytest.mark.slow
def test_calibration():

    rng = jax.random.PRNGKey(0)
    rng_logits, rng_labels, rng = jax.random.split(rng, 3)

    logits = jax.random.normal(rng_logits, (100, 2000, 12))
    labels = jax.random.randint(rng_labels, (100, 2000), 0, 12)
    label_probs = jax.nn.one_hot(labels, 12)

    cal_states, cal_models = calibrate.calibrate_all(logits, label_probs, cal_config)

    metrics = calibrate.calibrated_metrics(
        logits, labels, label_probs, cal_states, cal_models, metrics_fn
    )

    orig_ce = metrics[""]["ce_loss"].mean(-1)

    for cal_name in cal_states:
        new_ce = metrics[cal_name]["ce_loss"].mean(-1)

        if cal_name:
            assert jnp.all(new_ce < orig_ce)


@pytest.mark.slow
def test_single_class():

    logits = jax.random.uniform(jax.random.PRNGKey(0), (100, 2000, 12))
    labels = jnp.ones((100, 2000), dtype=jnp.int32)
    label_probs = jax.nn.one_hot(labels, 12)

    cal_states, cal_models = calibrate.calibrate_all(logits, label_probs, cal_config)

    metrics = calibrate.calibrated_metrics(
        logits, labels, label_probs, cal_states, cal_models, metrics_fn
    )

    orig_ce = metrics[""]["ce_loss"].mean(-1)

    for cal_name in cal_states:
        new_ce = metrics[cal_name]["ce_loss"].mean(-1)

        if "b" in cal_name or "v" in cal_name:
            assert jnp.all(new_ce < 1e-1)
        elif cal_name == "t":
            assert jnp.all(new_ce < orig_ce)
            assert jnp.all(new_ce > 1)
        else:
            assert jnp.all(new_ce == orig_ce)


@pytest.mark.slow
def test_uniform():
    logits = jax.nn.one_hot(jnp.zeros((100, 2000), dtype=jnp.int32), 12)
    logits = logits * jax.random.uniform(jax.random.PRNGKey(0), (100, 2000, 1)) * 20

    labels = jnp.zeros((100, 2000), dtype=jnp.int32)
    label_probs = jnp.ones((100, 2000, 12)) / 12

    cal_states, cal_models = calibrate.calibrate_all(logits, label_probs, cal_config)

    metrics = calibrate.calibrated_metrics(
        logits, labels, label_probs, cal_states, cal_models, metrics_fn
    )

    orig_ce = metrics[""]["ce_loss"].mean(-1)
    best_ce = jnp.log(12)

    for cal_name in cal_states:
        new_ce = metrics[cal_name]["ce_loss"].mean(-1)

        if "v" in cal_name or "t" in cal_name:
            assert jnp.all(jnp.mean(new_ce - best_ce) < 1e-1)
        elif cal_name == "b":
            assert jnp.all(new_ce < orig_ce)
            assert jnp.all(new_ce > best_ce + 1e-5)
        else:
            assert jnp.all(new_ce == orig_ce)
