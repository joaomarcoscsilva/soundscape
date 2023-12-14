import glob
import os
import pickle

import jax
import tensorflow as tf
from jax import numpy as jnp
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
    training,
    vit,
)
from soundscape.settings import settings_fn


@settings_fn
def get_soundscape_dataset(rng, *, batch_size):
    ds = (
        dataset.get_tensorflow_dataset("u", rng)
        .map(dataset.split_image)
        .unbatch()
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return ds


@settings_fn
def get_preprocess_functions():
    preprocess = dataset.prepare_image | dataset.downsample_image

    preprocess = dataset.tf2jax | composition.jit(preprocess)

    return preprocess


@settings_fn
def get_functions(rng, *, model_name):
    if "resnet" in model_name:
        call_fn, values = resnet.resnet(rng)
    elif "vit" in model_name:
        call_fn, values = vit.vit(rng)

    del values

    # call_fn = composition.vmap(call_fn, vmapped_keys = ('params', 'state'))
    call_fn = composition.jit(call_fn, static_keys=["is_training"])
    call_fn = call_fn | log.track(["logits", "id"])

    log_fn = log.stack_epoch_logs | log.save_logs

    return call_fn, log_fn


@settings_fn
def load_model_weights(*, weights_dir):
    params = []
    for weights_file in sorted(glob.glob(os.path.join(weights_dir, "*.pkl"))):
        with open(weights_file, "rb") as f:
            params.append(pickle.load(f))
    # params = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *params)
    return params


@settings_fn
def predict(*, seed, name):
    rng = jax.random.PRNGKey(seed)
    ds = get_soundscape_dataset(rng)

    preprocess = get_preprocess_functions()

    call_fn, log_fn = get_functions(rng)

    values = load_model_weights()

    for batch in tqdm(ds, position=1):
        processed_batch = preprocess(batch)
        for i in range(len(values)):
            v = call_fn({**values[i], **processed_batch, "is_training": False})
            values[i]["_logs"] = v["_logs"]

    for i in range(len(values)):
        values[i] = log.stack_epoch_logs(values[i])
        values[i] = values[i]["_epoch_logs"]

    values = log.merge_logs(values, "concat")

    with open(f"logs/{name}.pkl", "wb") as f:
        pickle.dump(values, f)


if __name__ == "__main__":
    with settings.Settings.from_command_line():
        predict()
