import os
import sys

if "--gpu" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from soundscape import settings, dataset
import time
from tqdm import tqdm
from jax import numpy as jnp
from tqdm import tqdm
import itertools
import jax
import pickle
import sys
import optax
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow_probability as tfp
import argparse
from jax.scipy.special import logsumexp


with settings.Settings.from_file("settings/supervised.yaml"):
    ws = dataset.get_class_weights()

@jax.jit
def balacc(logits, labels):
    probs = jax.nn.softmax(logits) * ws
    return ((labels == probs.argmax(1)) * ws[labels]).mean()


@jax.jit
def balacc_nobayes(logits, labels):
    probs = jax.nn.softmax(logits)
    return ((labels == probs.argmax(1)) * ws[labels]).mean()


@jax.jit
def transform(params, logits):
    logits = params["w"] * (logits) + params["b"]

    return logits


@jax.jit
def ce_loss(params, logits, one_hot_labels):
    transformed_logits = transform(params, logits)
    ce = optax.softmax_cross_entropy(transformed_logits, one_hot_labels).mean()
    return ce


@jax.jit
def entropy(probs):
    return -(probs * jnp.log(probs)).sum(axis=-1).mean()


def ECE(logits, labels, n_bins=5):
    return tfp.stats.expected_calibration_error(n_bins, logits, labels)


def calibrate(temperature_dim, bias_dim):
    @jax.jit
    def _calibrate(logits, labels):
        one_hot_labels = jax.nn.one_hot(labels, num_classes)
        params = {
            "w": jnp.ones(max(1, temperature_dim)),
            "b": jnp.zeros(bias_dim),
        }

        optim = optax.adam(1e-3)
        optim_state = optim.init(params)

        def update(params_and_optim_state, _):
            params, optim_state = params_and_optim_state
            grad_fn = jax.grad(ce_loss)
            grad = grad_fn(params, logits, one_hot_labels)
            updates, optim_state = optim.update(grad, optim_state)
            params = optax.apply_updates(params, updates)
            if temperature_dim == 0:
                params["w"] = jnp.ones(1)
            return ((params, optim_state), None)

        params, optim_state = jax.lax.scan(update, (params, optim_state), None, 10000)[
            0
        ]

        return params

    return _calibrate

def calibrate_all(logs_list):
    cal_scalar_nobias = jax.jit(jax.vmap(calibrate(1, 1)))
    cal_vector_nobias = jax.jit(jax.vmap(calibrate(num_classes, 1)))
    cal_notemp_bias = jax.jit(jax.vmap(calibrate(0, num_classes)))
    cal_scalar_bias = jax.jit(jax.vmap(calibrate(1, num_classes)))
    cal_vector_bias = jax.jit(jax.vmap(calibrate(num_classes, num_classes)))

    fns = {
        "scalar_nobias": cal_scalar_nobias,
        "vector_nobias": cal_vector_nobias,
        "notemp_bias": cal_notemp_bias,
        "scalar_bias": cal_scalar_bias,
        "vector_bias": cal_vector_bias,
    }

    for logs in logs_list:
        logs["calibration"] = {}

    for logs, (key, fn) in tqdm(
        itertools.product(logs_list, fns.items()), total=len(logs_list) * len(fns)
    ):
        logits = logs["logs"]["val_logits"]
        labels = logs["logs"]["val_one_hot_labels"].argmax(-1)

        logs["calibration"][key] = fn(logits, labels)

    return logs_list


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("files", nargs="+")
    argparser.add_argument("--gpu", action="store_true", default=False)
    argparser.add_argument("--target", default="cal")
    argparser.add_argument("--dataset", default="leec")
    args = argparser.parse_args()

    if args.dataset == "leec":
        with settings.Settings.from_file("settings/supervised.yaml"):
            ws = dataset.get_class_weights()
        num_classes = 12
    elif args.dataset == "leec2":
        with settings.Settings.from_file("settings/supervised.yaml"):
            ws = dataset.get_class_weights(num_classes=2)
        num_classes = 2
    else:
        with settings.Settings.from_file(f"settings/supervised.yaml"):
            with settings.Settings.from_file(f"settings/{args.dataset}.yaml"):
                ws = dataset.get_class_weights()
        num_classes = len(ws)

    for k in args.files:
        with open(k, "rb") as f:
            x = pickle.load(f)

        if not isinstance(x, list): 
            x = [{'logs':x}]

        x = calibrate_all(x)

        dirname = os.path.dirname(k)
        basename = os.path.basename(k)
        os.makedirs(f"{dirname}/{args.target}", exist_ok=True)

        with open(f"{dirname}/{args.target}/{basename}", "wb") as f:
            pickle.dump(x, f)
