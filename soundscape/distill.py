import os
import pickle
import sys

import jax
import numpy as np
import optax
import tensorflow as tf
from jax import numpy as jnp
from jax import random
from tqdm import tqdm

from soundscape import (
    augment,
    calibrate,
    composition,
    dataset,
    log,
    loss,
    resnet,
    settings,
    supervised,
    training,
    vit,
)
from soundscape.composition import Composable, identity
from soundscape.settings import settings_fn


@settings_fn
def get_u_dataset(rng, *, batch_size):
    ds = (
        dataset.get_distill_dataset(rng)
        .map(dataset.split_image)
        .unbatch()
        .shuffle(1000, seed=rng[0])
        .batch(batch_size)
    )

    return ds


@settings_fn
def get_u_preprocess_function(*, cutout_alpha, cutmix_alpha, mixup_alpha, crop_type):
    preprocess_u_train = (
        dataset.prepare_image
        | dataset.split_rng
        | dataset.downsample_image
        | augment.cutout(cutout_alpha)
        | augment.cutmix(cutmix_alpha)
        | augment.mixup(mixup_alpha)
    )

    preprocess_u_train = dataset.tf2jax | composition.jit(preprocess_u_train)

    return preprocess_u_train


@settings_fn
def get_model(rng, *, model_name):
    if "resnet" in model_name:
        call_fn, values = resnet.resnet(rng)
    elif "vit" in model_name:
        call_fn, values = vit.vit(rng)
    return call_fn, values


@settings_fn
def get_optimizer(
    values,
    steps_per_epoch,
    *,
    optim_name,
    log_learning_rate,
    epochs,
    sub_log_momentum,
    log_weight_decay,
):
    lr_schedule = optax.cosine_decay_schedule(
        init_value=10 ** (log_learning_rate),
        decay_steps=steps_per_epoch * epochs,
    )

    print("Total steps:", steps_per_epoch * epochs)

    if optim_name == "adam":
        base_optim_transform = optax.scale_by_adam()
    elif optim_name == "sgd":
        if sub_log_momentum < -0.99:
            base_optim_transform = optax.trace(decay=1 - 10**sub_log_momentum)
        else:
            base_optim_transform = optax.identity()

    if log_weight_decay < -0.99:
        weight_decay_transform = optax.add_decayed_weights(10**log_weight_decay)
    else:
        weight_decay_transform = optax.identity()

    optim = optax.chain(
        base_optim_transform,
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        weight_decay_transform,
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    values["optim_state"] = optim.init(values["params"])

    return optim, values


@settings_fn
def get_metrics_function(call_fn):
    weights = dataset.get_class_weights()

    metrics_fn = (
        call_fn
        | Composable(loss.mean(loss.crossentropy), "loss")
        | Composable(loss.preds(weights), "preds")
        | Composable(loss.preds(), "preds_nb")
        | Composable(loss.accuracy("preds"), "acc")
        | Composable(loss.accuracy("preds_nb"), "acc_nb")
        | Composable(loss.weighted(loss.accuracy("preds"), weights), "bal_acc")
        | Composable(loss.weighted(loss.accuracy("preds_nb"), weights), "bal_acc_nb")
        | Composable(loss.crossentropy, "ce_loss")
        | Composable(loss.brier, "brier")
    )

    return metrics_fn


@settings_fn
def get_call_functions(
    metrics_fn, optim, train_ds, val_ds, test_ds, *, evaluate_on_test, u_batches
):
    logged_keys = [
        "ce_loss",
        "brier",
        "acc",
        "acc_nb",
        "bal_acc",
        "bal_acc_nb",
        "epoch",
        "preds",
        "preds_nb",
        "logits",
        "one_hot_labels",
    ]
    pbar_keys = [
        "epoch",
        "ce_loss",
        "bal_acc",
        "val_ce_loss",
        "val_bal_acc",
        "val_acc_nb",
    ]
    pbar_every = 11
    pbar_len = (
        u_batches
        + len(train_ds)
        + len(val_ds)
        + (len(test_ds) if evaluate_on_test else 0)
    )

    evaluate_and_grad = composition.grad(metrics_fn, "params", "loss")
    train = evaluate_and_grad | training._get_update_fn(optim)

    train = composition.jit(train, static_keys=["is_training"])
    evaluate = composition.jit(metrics_fn, static_keys=["is_training"])

    track_progress = log.count_steps | log.track_progress(
        pbar_keys, pbar_every, pbar_len
    )

    train = train | log.track(logged_keys) | track_progress
    test = evaluate | log.track(logged_keys, prefix="test_") | track_progress
    evaluate = evaluate | log.track(logged_keys, prefix="val_") | track_progress

    return train, evaluate, test


@settings_fn
def get_log_function():
    mean_metrics = [
        "ce_loss",
        "brier",
        "val_ce_loss",
        "val_brier",
        "bal_acc",
        "val_bal_acc",
        "bal_acc_nb",
        "val_bal_acc_nb",
    ]

    log_function = (
        log.stack_epoch_logs
        | log.mean_over_epoch(mean_metrics)
        | log.save_logs
        | log.save_params
        | log.stop_if_nan(mean_metrics)
        | log.stop_if_no_improvement()
    )

    return log_function


@settings_fn
def train(*, training_seed, epochs, name, evaluate_on_test, u_batches):
    print(f"Training {name} for {epochs} epochs")

    rng = random.PRNGKey(training_seed)
    rng_ds, rng_net, rng = random.split(rng, 3)

    train_ds, val_ds, test_ds = supervised.get_soundscape_dataset(rng_ds)
    train_u_ds = get_u_dataset(rng_ds)

    breakpoint()

    preprocess_train, preprocess_val = supervised.get_preprocess_functions()
    preprocess_u_train = get_u_preprocess_function()

    call_fn, values = get_model(rng_net)
    optim, values = get_optimizer(values, len(train_ds) + u_batches)
    metrics_fn = get_metrics_function(call_fn)
    train, evaluate, test = get_call_functions(
        metrics_fn, optim, train_ds, val_ds, test_ds
    )

    log_function = get_log_function()

    values["rng"] = rng

    for e in tqdm(range(epochs), position=1, ncols=140, smoothing=0.9):
        values["epoch"] = jnp.array([e + 1])

        for batch in train_u_ds.take(u_batches):
            values = preprocess_u_train({**values, **batch})
            values = train({**values, "is_training": True})

        for batch in train_ds:
            values = preprocess_train({**values, **batch})
            values = train({**values, "is_training": True})

        for batch in val_ds:
            values = preprocess_val({**values, **batch})
            values = evaluate({**values, "is_training": False})

        if evaluate_on_test:
            old_rng = values["rng"]

            for batch in test_ds:
                values = preprocess_val({**values, **batch})
                values = test({**values, "is_training": False})

            values["rng"] = old_rng

        values = log_function(values)

        if "_stop" in values:
            print("Early stopping:", values["_stop"])
            values.pop("_stop")
            break

    return values["_epoch_logs"]


if __name__ == "__main__":
    with settings.Settings.from_command_line():
        train()
