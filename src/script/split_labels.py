import pandas as pd
import sys
import os
import jax

from soundscape.lib.settings import settings

rng = jax.random.PRNGKey(0)

df = pd.read_csv(os.path.join(settings["data"]["data_dir"], "labels.csv"))

test_frac = settings["data"]["split"]["test_frac"]
val_frac = settings["data"]["split"]["val_frac"]
train_frac = 1 - test_frac - val_frac


files = df["file"].unique()
n = len(files)
idx = jax.random.permutation(rng, jax.numpy.arange(n))
files = files[idx]

test_files = files[: int(n * test_frac)]
val_files = files[int(n * test_frac) : int(n * (test_frac + val_frac))]
train_files = files[int(n * (test_frac + val_frac)) :]

test_df = df[df["file"].isin(test_files)]
val_df = df[df["file"].isin(val_files)]
train_df = df[df["file"].isin(train_files)]

train_df.to_csv(
    os.path.join(settings["data"]["data_dir"], "train_labels.csv"),
    index=False,
)
val_df.to_csv(
    os.path.join(settings["data"]["data_dir"], "val_labels.csv"),
    index=False,
)
test_df.to_csv(
    os.path.join(settings["data"]["data_dir"], "test_labels.csv"),
    index=False,
)
