import jax
import optax

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
        print(i, cal_state.params["w"][:2])
        cal_outputs, cal_state = optimizing.update(
            cal_batch, cal_state, cal_model, cal_optim
        )

    return ModelState(cal_state.params, None, None, None)


def calibrate_all(logits, label_probs, lr, num_steps):
    num_classes = logits.shape[-1]

    w_shapes = [0, 1, num_classes]
    b_shapes = [0, num_classes]

    w_names = ["", "t", "v"]
    b_names = ["", "b"]

    all_cal_states = {}
    all_cal_models = {}

    for w_name, w_dim in zip(w_names, w_shapes):
        for b_name, b_dim in zip(b_names, b_shapes):

            cal_model, cal_state, cal_optim = _instantiate_calibration(
                logits.shape[0], w_dim, b_dim, lr
            )
            cal_state = calibrate(
                logits, label_probs, num_steps, cal_model, cal_state, cal_optim
            )

            name = w_name + b_name
            all_cal_states[w_name + b_name] = cal_state
            all_cal_models[name] = cal_model

    return all_cal_states, all_cal_models
