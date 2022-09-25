import jax
import os

from soundscape.data import dataset
from soundscape.lib import model, supervised
from soundscape.lib.settings import settings


if not os.path.exists("models"):
    os.mkdir("models")

if not os.path.exists("plots"):
    os.mkdir("plots")


rng = jax.random.PRNGKey(settings["seed"])

train_ds = dataset.spectrogram_dataset(
    settings, from_disk=True, path="train_labels.csv"
).cache()


val_settings = settings.copy()
val_settings["data"]["fragmentation"]["begin_time_fn"] = "fixed"

val_ds = dataset.spectrogram_dataset(
    val_settings, from_disk=True, path="val_labels.csv"
).cache()

test_ds = dataset.spectrogram_dataset(
    val_settings, from_disk=True, path="test_labels.csv"
).cache()

supervised.train(settings, rng, train_ds, val_ds=val_ds, model_fn=model.resnet)
