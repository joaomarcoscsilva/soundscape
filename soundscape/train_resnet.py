from jax import random, numpy as jnp
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
from soundscape.composition import Composable

settings_fn, settings_dict = settings.from_file("model_settings.yaml")

rng = random.PRNGKey(0)
rng_ds, rng_net, rng = random.split(rng, 3)

ds = dataset.get_tensorflow_dataset("train", rng_ds).batch(32)

call_fn, params = resnet.resnet(rng_net)

optim = optax.adam(1e-3)
params["optim_state"] = optim.init(params["params"])

process_batch = (
    dataset.tf2jax
    | dataset.prepare_image
    | dataset.prepare_image_channels
    | dataset.downsample_image
    | dataset.one_hot_encode
)

predict = (
    call_fn
    | Composable(loss.mean(loss.crossentropy), "loss")
    | Composable(loss.preds, "preds")
    | Composable(loss.crossentropy, "ce_loss")
    | Composable(loss.accuracy, "acc")
)

train = composition.grad(predict, "params", "loss") | training.update(optax.adam(1e-3))
train = composition.jit(
    train, static_keys=["is_training"], ignored_keys=["file", "tqdm"]
)
train = (
    train | log.track(["ce_loss", "acc"]) | log.log_tqdm(["ce_loss", "acc"], len(ds))
)

for e in range(settings_dict["epochs"]):
    for batch in ds:
        batch = process_batch(batch)
        params = train({**params, **batch, "is_training": True})
