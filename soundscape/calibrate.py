import jax
import optax
from jax import numpy as jnp

from soundscape import composition, settings, supervised, training
from soundscape.composition import Composable


@Composable
def calibrate_transform(values):
    calibration_params = values["calibration_params"]
    logits = values["logits"]
    return {
        **values,
        "calibrated_logits": calibration_params["w"] * logits + calibration_params["b"],
    }


@settings.settings_fn
def calibrate(
    metrics_fn,
    prefix,
    *,
    calibration_steps,
    calibration_lr,
    calibration_optim,
    calibration_type,
    num_classes,
):
    if calibration_optim == "sgd":
        optim = optax.sgd(calibration_lr)
    elif calibration_optim == "adam":
        optim = optax.adam(calibration_lr)
    else:
        raise ValueError(f"Unknown calibration_optim: {calibration_optim}")

    if calibration_type == "linear":
        initial_calibration_params = {"w": jnp.ones(1), "b": jnp.zeros(1)}
    elif calibration_type == "affine":
        initial_calibration_params = {"w": jnp.ones(1), "b": jnp.zeros(num_classes)}
    elif calibration_type == "full":
        initial_calibration_params = {
            "w": jnp.ones(num_classes),
            "b": jnp.zeros(num_classes),
        }

    @Composable
    def _calibrate(values):
        calibration_params = initial_calibration_params

        evaluate_fn = calibrate_transform | metrics_fn
        grad_fn = composition.grad(evaluate_fn, "calibration_params", "loss")
        step_fn = grad_fn | training._get_update_fn(optim)

        step_fn = composition.jit(step_fn)

        cal_values = {
            "calibration_params": calibration_params,
            "logits": cal_values["val_logits"][-1],
            "labels": cal_values["val_labels"][-1],
        }

        for _ in range(calibration_steps):
            cal_values = step_fn(cal_values)
