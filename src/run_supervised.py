import jax
import os

from soundscape.data import dataset
from soundscape.lib import model, supervised
from soundscape.lib.settings import settings

os.mkdir("models")
os.mkdir("plots")

rng = jax.random.PRNGKey(settings["seed"])

train_ds = dataset.spectrogram_dataset(from_disk=True, path="train_labels.csv").cache()
val_ds = dataset.spectrogram_dataset(from_disk=True, path="val_labels.csv").cache()
test_ds = dataset.spectrogram_dataset(from_disk=True, path="test_labels.csv").cache()

supervised.train(rng, train_ds, val_ds=val_ds, model_fn=model.resnet)
