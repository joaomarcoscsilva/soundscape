import jax
import optax
from tqdm import tqdm

from . import optimizing
from .models.base_model import ModelState
from .models.calibrator import calibrator


def _instantiate_calibration(num_epochs, w_dim, b_dim, lr):
    cal_model, cal_state = calibrator(num_epochs, w_dim, b_dim)

    cal_optim = optax.adam(lr)
    optim_state = cal_optim.init(cal_state.params)
    cal_state = cal_state._replace(optim_state=optim_state)

    return cal_model, cal_state, cal_optim


def calibrate(logits, label_probs, num_steps, cal_model, cal_state, cal_optim):

    cal_batch = {"inputs": logits, "label_probs": label_probs}

    for i in range(num_steps):
        cal_outputs, cal_state = optimizing.update(
            cal_batch, cal_state, cal_model, cal_optim
        )

    return ModelState(params=cal_state.params)


def calibrate_all(logits, label_probs, cal_config):
    num_classes = logits.shape[-1]

    w_shapes = [0, 1, num_classes]
    b_shapes = [0, num_classes]

    w_names = ["", "t", "v"]
    b_names = ["", "b"]

    all_cal_states = {}
    all_cal_models = {}

    tqdm_bar = tqdm(total=len(w_shapes) * len(b_shapes) - 1)
    for w_name, w_dim in zip(w_names, w_shapes):
        for b_name, b_dim in zip(b_names, b_shapes):

            cal_model, cal_state, cal_optim = _instantiate_calibration(
                logits.shape[0], w_dim, b_dim, cal_config["lr"]
            )
            cal_state = calibrate(
                logits,
                label_probs,
                cal_config["num_steps"],
                cal_model,
                cal_state,
                cal_optim,
            )

            name = w_name + b_name
            all_cal_states[name] = cal_state
            all_cal_models[name] = cal_model

            tqdm_bar.update()

    return all_cal_states, all_cal_models


def calibrated_metrics(logits, labels, label_probs, cal_states, cal_models, metrics_fn):
    def _calibrated_metrics(cal_state, cal_model):
        cal_logits = cal_model(logits, cal_state, False)[0]["logits"]

        return metrics_fn(
            {"labels": labels, "label_probs": label_probs},
            {"logits": cal_logits},
        )

    return {
        cal_name: _calibrated_metrics(cal_states[cal_name], cal_models[cal_name])
        for cal_name in cal_states
    }


def get_calibrated_metrics(logger, cal_states, cal_models, metrics_fn):
    splits = ["val", "test"]
    splits = [split for split in splits if split + "_logits" in logger.merge()]

    metrics = {split: {} for split in splits}
    best_metrics = {split: {} for split in splits}

    for split in splits:
        logits = logger[split + "_logits"]
        labels = logger[split + "_labels"]

        label_probs = (
            logger[split + "_label_probs"]
            if split + "_label_probs" in logger.merge()
            else jax.nn.one_hot(labels, logits.shape[-1])
        )

        metrics[split] = calibrated_metrics(
            logits, labels, label_probs, cal_states, cal_models, metrics_fn
        )

    for split in splits:

        best_metrics[split] = {
            cal_name: logger.best_epoch_metrics(
                metrics["val"][cal_name], metrics[split][cal_name], prefix="val_"
            )
            for cal_name in cal_states
        }

    return best_metrics
