import glob
import os

import hydra
import jax
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from jax import numpy as jnp
from jax import random

from soundscape.dataset import dataloading, dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CLASSES = [
    "basi_culi",
    "myio_leuc",
    "vire_chiv",
    "cycl_guja",
    "pita_sulp",
    "zono_cape",
    "dend_minu",
    "apla_leuc",
    "isch_guen",
    "phys_cuvi",
    "boan_albo",
    "aden_marm",
]


def class_to_binary(class_):
    """
    Convert a class from the 13-class dataset to a binary class indicating whether
    the corresponding species is a bird or a frog.
    """

    return 0 if CLASSES.index(class_) < 6 else 1


def get_ds():
    with hydra.initialize("../../new_settings", version_base=None):
        settings = hydra.compose(config_name="dataset/leec")
        ds = hydra.utils.instantiate(settings.dataset)
    return ds


def get_splits_metadata(ds, **kwargs):
    rng = random.PRNGKey(0)

    train_metadata = dataloading.load_split_metadata(
        rng, "data/leec/train", ds, **kwargs
    )
    val_metadata = dataloading.load_split_metadata(rng, "data/leec/val", ds, **kwargs)
    test_metadata = dataloading.load_split_metadata(rng, "data/leec/test", ds, **kwargs)

    return train_metadata, val_metadata, test_metadata


def assert_split_size(split_metadata, size):
    for v in split_metadata.values():
        assert len(v) == size


def assert_disjoint(*splits_metadata):
    for i in range(len(splits_metadata)):
        for j in range(i + 1, len(splits_metadata)):
            i_ids = set(splits_metadata[i]["_ids"])
            j_ids = set(splits_metadata[j]["_ids"])
            assert i_ids & j_ids == set()


def assert_has_all_classes(*splits_metadata, num_classes):
    for split_metadata in splits_metadata:
        assert len(set(split_metadata["labels"])) == num_classes


def assert_no_repeats(*splits_metadata):
    for split_metadata in splits_metadata:
        assert len(set(split_metadata["_ids"])) == len(split_metadata["_ids"])


def assert_roughly_balanced(*splits_metadata):
    counts = []
    for split_metadata in splits_metadata:
        counts.append(
            np.bincount(split_metadata["labels"]) / len(split_metadata["labels"])
        )
    counts = np.array(counts)
    assert np.allclose(counts.std(axis=0), 0, atol=0.05)


def test_leec12_sizes():
    ds = get_ds()

    # Test that the number of files matches the 60/20/20 split
    train_size = 2293
    val_size = test_size = 758

    train_metadata, val_metadata, test_metadata = get_splits_metadata(
        ds, class_to_id=ds.class_order.index, skipped_classes=["other"]
    )

    assert_split_size(train_metadata, train_size)
    assert_split_size(val_metadata, val_size)
    assert_split_size(test_metadata, test_size)

    assert_disjoint(train_metadata, val_metadata, test_metadata)
    assert_has_all_classes(train_metadata, val_metadata, test_metadata, num_classes=12)
    assert_no_repeats(train_metadata, val_metadata, test_metadata)
    assert_roughly_balanced(train_metadata, val_metadata, test_metadata)


def test_leec13_sizes():
    ds = get_ds()

    # Test that the number of files matches the 60/20/20 split
    train_size = 9990
    val_size = test_size = 3323

    train_metadata, val_metadata, test_metadata = get_splits_metadata(
        ds, class_to_id=ds.class_order.index
    )

    assert_split_size(train_metadata, train_size)
    assert_split_size(val_metadata, val_size)
    assert_split_size(test_metadata, test_size)

    assert_disjoint(train_metadata, val_metadata, test_metadata)
    assert_has_all_classes(train_metadata, val_metadata, test_metadata, num_classes=13)
    assert_no_repeats(train_metadata, val_metadata, test_metadata)
    assert_roughly_balanced(train_metadata, val_metadata, test_metadata)


def test_leec2_sizes():
    ds = get_ds()

    # Test that the number of files matches the 60/20/20 split
    train_size = 2293
    val_size = test_size = 758

    train_metadata, val_metadata, test_metadata = get_splits_metadata(
        ds, class_to_id=class_to_binary, skipped_classes=["other"]
    )

    assert_split_size(train_metadata, train_size)
    assert_split_size(val_metadata, val_size)
    assert_split_size(test_metadata, test_size)

    assert_disjoint(train_metadata, val_metadata, test_metadata)
    assert_has_all_classes(train_metadata, val_metadata, test_metadata, num_classes=2)
    assert_no_repeats(train_metadata, val_metadata, test_metadata)
    assert_roughly_balanced(train_metadata, val_metadata, test_metadata)


def test_leec_dataloader():
    ds = get_ds()
    rng = jax.random.PRNGKey(0)

    dataloader = dataloading.DataLoader(
        rng,
        ds,
        "image",
        class_to_id=ds.class_order.index,
        cache=False,
        skipped_classes=["other"],
    )

    batches = []
    for i, batch in enumerate(dataloader.iterate(rng, "train", 32)):
        batches.append(batch)

        assert batch["inputs"].shape == (32, 256, 423, 1)
        assert batch["labels"].shape == (32,)
        assert batch["rngs"].shape[0] == 32
        assert batch["rng"].shape == (2,)
        assert isinstance(batch["inputs"], jax.Array)

        if i == 10:
            break

    dataloader = dataloading.DataLoader(
        rng,
        ds,
        "image",
        class_to_id=ds.class_order.index,
        cache=True,
        skipped_classes=["other"],
    )

    for i, batch in enumerate(dataloader.iterate(rng, "train", 32)):
        assert (batch["inputs"] == batches[i]["inputs"]).all()
        assert (batch["labels"] == batches[i]["labels"]).all()
        assert (batch["rngs"] == batches[i]["rngs"]).all()
        assert (batch["rng"] == batches[i]["rng"]).all()

        if i == 10:
            break


def test_tf2jax():
    a = {
        "inputs": tf.random.uniform((10, 256, 256, 3)),
        "labels": tf.random.uniform((10,), minval=0, maxval=13, dtype=tf.int32),
        "id": tf.random.uniform((10,), minval=0, maxval=1000, dtype=tf.int32),
        "rngs": tf.random.uniform((10, 2), minval=0, maxval=1000, dtype=tf.int32),
        "_file": tf.constant(["string"] * 10),
    }

    b = dataloading.tf2jax(a)

    assert isinstance(b["inputs"], jnp.ndarray)
    assert isinstance(b["labels"], jnp.ndarray)
    assert isinstance(b["id"], jnp.ndarray)
    assert isinstance(b["rngs"], jnp.ndarray)
    assert type(b["_file"]) == np.ndarray

    assert (b["inputs"] == a["inputs"].numpy()).all()
    assert (b["labels"] == a["labels"].numpy()).all()
    assert (b["id"] == a["id"].numpy()).all()
    assert (b["rngs"] == a["rngs"].numpy()).all()
    assert (b["_file"] == a["_file"].numpy()).all()
