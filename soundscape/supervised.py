from jax import random, numpy as jnp
import numpy as np
from tqdm import tqdm
import optax
import pickle
import sys
import os

from soundscape import (
    dataset,
    augment,
    resnet,
    vit,
    loss,
    training,
    composition,
    settings,
    log,
)
from composition import Composable, identity

if len(sys.argv) > 2:
    for i, arg in enumerate(sys.argv[1:]):
        print(f"\n\nRunning experiment {i+1} of {len(sys.argv[1:])} ({arg})\n\n")
        errcode = os.system(f"python {sys.argv[0]} {arg}")
        if errcode:
            print("\n Error code: ", errcode)
            sys.exit(errcode)
    sys.exit(0)


"""
Setup
"""

settings_fn, settings_dict = settings.from_file()

rng = random.PRNGKey(0)
rng_ds, rng_net, rng = random.split(rng, 3)
"""
Dataset
"""

rng_train_ds, rng_val_ds = random.split(rng_ds)
ds = (
    dataset.get_tensorflow_dataset("train", rng_train_ds)
    .shuffle(1000, seed=0)
    .batch(settings_dict["batch_size"])
)
val_ds = (
    dataset.get_tensorflow_dataset("val", rng_val_ds)
    .shuffle(1000, seed=0)
    .batch(settings_dict["batch_size"])
)


process_train_batch = (
    dataset.prepare_image
    | dataset.one_hot_encode
    | augment.time_crop()
    | dataset.downsample_image
    | augment.cutout(settings_dict["cutout_alpha"])
    | augment.cutmix(settings_dict["cutmix_alpha"])
    | augment.mixup(settings_dict["mixup_alpha"])
)

process_train_batch = composition.jit(process_train_batch)
process_train_batch = dataset.tf2jax | process_train_batch

process_val_batch = (
    dataset.prepare_image
    | dataset.one_hot_encode
    | augment.time_crop(crop_type="deterministic")
    | dataset.downsample_image
)

process_val_batch = composition.jit(process_val_batch)
process_val_batch = dataset.tf2jax | process_val_batch

weights = dataset.get_class_weights()

"""
Network and Optimizer
"""

if "resnet" in settings_dict["model_name"]:
    call_fn, params = resnet.resnet(rng_net)
elif "vit" in settings_dict["model_name"]:
    call_fn, params = vit.vit(rng_net)

# lr_schedule = optax.warmup_cosine_decay_schedule(
#     init_value=0.0,
#     peak_value=settings_dict["learning_rate"],
#     warmup_steps=len(ds),
#     decay_steps=len(ds) * (settings_dict["epochs"] - 1),
# )

lr_schedule = optax.cosine_decay_schedule(
    init_value=settings_dict["learning_rate"],
    decay_steps=len(ds) * settings_dict["epochs"],
)

optim = optax.sgd(lr_schedule, momentum=0.9)

params["optim_state"] = optim.init(params["params"])

"""
Predict Function
"""

predict = (
    call_fn
    | Composable(loss.mean(loss.crossentropy), "loss")
    | Composable(loss.preds, "preds")
    | Composable(loss.crossentropy, "ce_loss")
    | Composable(loss.accuracy, "acc")
    | Composable(loss.weighted(loss.accuracy, weights), "bal_acc")
    | Composable(loss.weighted(loss.crossentropy, weights), "bal_ce_loss")
)

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
    "one_hot_labels"
]

"""
Training Function
"""

train = composition.grad(predict, "params", "loss") | training.update(optim)
train = composition.jit(train, static_keys=["is_training"])
train = train | log.track(logged_keys)

"""
Evaluation Function
"""

evaluate = composition.jit(predict, static_keys=["is_training"])
evaluate = evaluate | log.track(logged_keys, prefix="val_")

track_progress = log.track_progress(
    ["epoch", "ce_loss", "bal_acc", "val_ce_loss", "val_bal_acc"],
    every=5,
    total=len(ds) + len(val_ds),
)

train = train | track_progress
evaluate = evaluate | track_progress

"""
Training
"""

params_keys = params.keys()

print("Training...\n\n\n")

logs = []

for e in tqdm(range(settings_dict["epochs"]), position=1, ncols=140, smoothing=0.9):

    values = {**params, "is_training": True, "epoch": jnp.array([e + 1])}
    del params

    for batch in ds:
        batch = process_train_batch(batch)
        values = train({**values, **batch})

    values["is_training"] = True

    for batch in val_ds:
        batch = process_val_batch(batch)
        values = evaluate({**values, **batch})

    params = {k: values[k] for k in params_keys}
    logs.append(values["_logs"])

    filename = sys.argv[1].replace(".yaml", ".pkl").replace("settings/", "logs/")

    with open(filename, "wb") as f:
        pickle.dump(logs, f)
