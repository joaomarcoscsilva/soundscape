from jax import random, numpy as jnp
import jax
from tqdm import tqdm
import optax

from soundscape import (
    dataset,
    augment,
    resnet,
    loss,
    training,
    composition,
    settings,
    log,
)

"""
Setup
"""

settings_fn, settings_dict = settings.from_file("model_settings.yaml")

rng = random.PRNGKey(0)
rng_ds, rng_net, rng = random.split(rng, 3)

"""
Dataset
"""

ds = dataset.get_tensorflow_dataset("train", rng_ds).batch(32)

process_batch = (
    dataset.prepare_image
    | dataset.prepare_image_channels
    | dataset.downsample_image
    | dataset.one_hot_encode
)

process_batch = composition.jit(process_batch, ignored_keys=["file"])
process_batch = dataset.tf2jax | process_batch

class_weights = dataset.get_class_weights()

"""
Network and Optimizer
"""

call_fn, params = resnet.resnet(rng_net)

optim = optax.adamw(1e-3)

params["optim_state"] = optim.init(params["params"])

"""
Predict Function
"""

predict = (
    call_fn
    | composition.Composable(loss.mean(loss.crossentropy), "loss")
    | composition.Composable(loss.preds, "preds")
    | composition.Composable(loss.crossentropy, "ce_loss")
    | composition.Composable(loss.accuracy, "acc")
)

"""
Training Function
"""

train = composition.grad(predict, "params", "loss") | training.update(optax.adam(1e-3))
train = composition.jit(train, static_keys=["is_training"], ignored_keys=["file"])
train = (
    train
    | log.track(["ce_loss", "acc"])
    | log.track_progress(["ce_loss", "acc"], every=10)
)

"""
Training
"""

for e in range(settings_dict["epochs"]):
    values = {**params, "is_training": True}

    for batch in ds:
        batch = process_batch(batch)
        values = train({**values, **batch})

    logs = values["_logs"]
    params = {k: values[k] for k in params.keys()}
