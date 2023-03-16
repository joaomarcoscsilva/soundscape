from jax import random, numpy as jnp
import tensorflow as tf
import jax
import numpy as np
from tqdm import tqdm
import optax
import pickle
import sys
import os

from soundscape import (
    augment,
    dataset,
    resnet,
    vit,
    loss,
    training,
    composition,
    settings,
    log,
)

from soundscape.composition import Composable, identity
from soundscape.settings import settings_fn


@settings_fn
def get_soundscape_dataset(rng, *, batch_size):
    rng_train_ds, rng_val_ds, rng_test_ds = random.split(rng, 3)

    ds = (
        dataset.get_tensorflow_dataset("train", rng_train_ds)
        .cache()
        .shuffle(100, seed=rng_train_ds[0])
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    val_ds = (
        dataset.get_tensorflow_dataset("val", rng_val_ds)
        .shuffle(100, seed=rng_val_ds[0])
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_ds = (
        dataset.get_tensorflow_dataset("test", rng_test_ds)
        .shuffle(100, seed=rng_test_ds[0])
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return ds, val_ds, test_ds


@settings_fn
def get_preprocess_functions(*, cutout_alpha, cutmix_alpha, mixup_alpha, crop_type):
    preprocess_train = (
        dataset.prepare_image
        | dataset.one_hot_encode
        | dataset.split_rng
        | augment.time_crop(crop_type=crop_type)
        | dataset.downsample_image
        | augment.cutout(cutout_alpha)
        | augment.cutmix(cutmix_alpha)
        | augment.mixup(mixup_alpha)
    )

    preprocess_val = (
        dataset.prepare_image
        | dataset.one_hot_encode
        | augment.time_crop(
            crop_type="deterministic" if crop_type == "random" else crop_type
        )
        | dataset.downsample_image
    )

    preprocess_train = dataset.tf2jax | composition.jit(preprocess_train)
    preprocess_val = dataset.tf2jax | composition.jit(preprocess_val)

    return preprocess_train, preprocess_val


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
def get_call_functions(call_fn, optim, pbar_len):
    logged_keys = [
        "ce_loss",
        "acc",
        "bal_ce_loss",
        "bal_acc",
        "epoch",
        "id",
        "labels",
        "preds",
        "logits",
        "one_hot_labels",
    ]
    pbar_keys = ["epoch", "ce_loss", "bal_acc", "val_ce_loss", "val_bal_acc"]
    pbar_every = 11

    weights = dataset.get_class_weights()

    evaluate = (
        call_fn
        | Composable(loss.mean(loss.crossentropy), "loss")
        | Composable(loss.preds(weights), "preds")
        | Composable(loss.crossentropy, "ce_loss")
        | Composable(loss.accuracy, "acc")
        | Composable(loss.weighted(loss.accuracy, weights), "bal_acc")
        | Composable(loss.weighted(loss.crossentropy, weights), "bal_ce_loss")
    )

    evaluate_and_grad = composition.grad(evaluate, "params", "loss")
    train = evaluate_and_grad | training.update(optim)

    train = composition.jit(train, static_keys=["is_training"])
    evaluate = composition.jit(evaluate, static_keys=["is_training"])

    track_progress = log.count_steps | log.track_progress(
        pbar_keys, pbar_every, pbar_len
    )

    train = train | log.track(logged_keys) | track_progress
    evaluate = evaluate | log.track(logged_keys, prefix="val_") | track_progress

    return train, evaluate


@settings_fn
def get_log_function():
    log_function = (
        log.stack_epoch_logs
        | log.mean_over_epoch(["ce_loss", "bal_acc", "val_ce_loss", "val_bal_acc"])
        | log.save_logs
        | log.save_params
    )

    return log_function


@settings_fn
def train(
    *,
    training_seed,
    epochs,
    name,
    optimizing_metric,
    optimizing_mode,
    give_up_after,
    give_up_threshold,
):
    print(f"Training {name} for {epochs} epochs")

    rng = random.PRNGKey(training_seed)
    rng_ds, rng_net, rng = random.split(rng, 3)

    train_ds, val_ds, test_ds = get_soundscape_dataset(rng_ds)

    preprocess_train, preprocess_val = get_preprocess_functions()
    call_fn, values = get_model(rng_net)
    optim, values = get_optimizer(values, len(train_ds))
    train, evaluate = get_call_functions(call_fn, optim, len(train_ds) + len(val_ds))
    log_function = get_log_function()

    values["rng"] = rng

    for e in tqdm(range(epochs), position=1, ncols=140, smoothing=0.9):
        values["epoch"] = jnp.array([e + 1])

        for batch in train_ds:
            values = preprocess_train({**values, **batch})
            values = train({**values, "is_training": True})

        for batch in val_ds:
            values = preprocess_val({**values, **batch})
            values = evaluate({**values, "is_training": False})

        values = log_function(values)

        for metric in [
            "mean_ce_loss",
            "mean_bal_acc",
            "mean_val_ce_loss",
            "mean_val_bal_acc",
        ]:
            if metric in values["_epoch_logs"]:
                if jnp.isnan(values["_epoch_logs"][metric][-1]):
                    print(f"Early stopping at epoch {e} because of a NaN in {metric}")
                    return values["_epoch_logs"]

        if (e + 1) % give_up_after == 0:
            sign = 1 if optimizing_mode == "max" else -1
            last_metric = values["_epoch_logs"][optimizing_metric][-1]
            if sign * last_metric < sign * give_up_threshold:
                print(
                    f"Early stopping at epoch {e} because of slow convergence: {optimizing_metric} = {last_metric}"
                )
                return values["_epoch_logs"]

    return values["_epoch_logs"]


if __name__ == "__main__":
    with settings.Settings.from_command_line():
        train()
