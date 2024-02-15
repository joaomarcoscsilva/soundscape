import os

import hydra
import jax
import numpy as np
import pytest
import tensorflow as tf
from jax import numpy as jnp
from jax import random

from soundscape.dataset import dataloading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TESTING_BATCHES = 5

LEEC_SIZES = {
    "leec13": (9990, 3323, 3323),
    "leec12": (2293, 758, 758),
    "leec2": (2293, 758, 758),
}


def get_dataloader(name, include_test=True, **kwargs):
    rng = random.PRNGKey(0)
    with hydra.initialize("../../settings", version_base=None):
        settings = hydra.compose(config_name=f"dataloader/{name}")

    settings.dataloader.batch_size = 32
    return dataloading.get_dataloader(
        rng, settings.dataloader, include_test=include_test, **kwargs
    )


def assert_no_leaks(*metadata_sets):
    file_sets = [set(metadata["_files"]) for metadata in metadata_sets]
    id_sets = [set(metadata["_ids"]) for metadata in metadata_sets]

    assert sum(len(s) for s in file_sets) == len(set.union(*file_sets))
    assert sum(len(s) for s in id_sets) == len(set.union(*id_sets))


def get_splits_metadata(dataloader):
    rng = random.PRNGKey(0)

    train_metadata = dataloader.load_split_metadata(rng, "data/leec/train")
    val_metadata = dataloader.load_split_metadata(rng, "data/leec/val")
    test_metadata = dataloader.load_split_metadata(rng, "data/leec/test")

    assert_no_leaks(train_metadata, val_metadata, test_metadata)

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


def assert_metadata_sizes(dataloader, num_classes, train_size, val_size, test_size):
    train_metadata, val_metadata, test_metadata = get_splits_metadata(dataloader)

    assert_split_size(train_metadata, train_size)
    assert_split_size(val_metadata, val_size)
    assert_split_size(test_metadata, test_size)

    assert_disjoint(train_metadata, val_metadata, test_metadata)
    assert_has_all_classes(
        train_metadata, val_metadata, test_metadata, num_classes=num_classes
    )
    assert_no_repeats(train_metadata, val_metadata, test_metadata)
    assert_roughly_balanced(train_metadata, val_metadata, test_metadata)


def assert_dl_len(dl, sizes):
    assert len(dl) == (
        sizes[0] // dl.batch_size
        + jnp.ceil(sizes[1] / dl.batch_size)
        + jnp.ceil(sizes[2] / dl.batch_size)
    )


@pytest.mark.parametrize("num_classes", [13, 13, 2])
def test_leec_sizes(num_classes):
    dl_name = "leec" + str(num_classes)
    dl = get_dataloader(dl_name)
    assert_metadata_sizes(dl, num_classes, *LEEC_SIZES[dl_name])
    assert_dl_len(dl, LEEC_SIZES[dl_name])


@pytest.mark.parametrize("num_classes", [13, 13, 2])
def test_no_test_set(num_classes):
    dl_name = "leec" + str(num_classes)
    dl = get_dataloader(dl_name, include_test=False)

    assert set(dl._dataloaders.keys()) == {"train", "val"}
    assert_dl_len(dl, LEEC_SIZES[dl_name][:2] + (0,))


@pytest.mark.parametrize("dl_name", ["leec13", "leec12", "leec2"])
def test_dataloader_shapes(dl_name):
    rng = random.PRNGKey(0)
    dataloader = get_dataloader(dl_name)

    batches = []
    for i, batch in enumerate(dataloader.iterate(rng, "val")):
        batches.append(batch)

        assert batch["inputs"].shape == (32, 256, 423, 1)
        assert batch["labels"].shape == (32,)
        assert batch["rngs"].shape[0] == 32
        assert batch["rng"].shape == (2,)
        assert isinstance(batch["inputs"], jax.Array)

        if i == TESTING_BATCHES:
            break


def assert_equal_batch(batch1, batch2):
    assert (batch1["inputs"] == batch2["inputs"]).all()
    assert (batch1["labels"] == batch2["labels"]).all()
    assert (batch1["rngs"] == batch2["rngs"]).all()
    assert (batch1["rng"] == batch2["rng"]).all()


@pytest.mark.parametrize("dl_name", ["leec13", "leec12", "leec2"])
def test_reproducible_dataloaders(dl_name):
    rng1 = random.PRNGKey(0)
    batches = []

    dataloader = get_dataloader(dl_name)
    for i, batch in enumerate(dataloader.iterate(rng1, "val")):
        batches.append(batch)
        if i == 10:
            break

    dataloader = get_dataloader(dl_name)
    for i, batch in enumerate(dataloader.iterate(rng1, "val")):
        assert_equal_batch(batch, batches[i])
        if i == 10:
            break


def test_prior_weights_leec13():
    dl = get_dataloader("leec13")
    weights = dl.prior_weights()

    assert weights.min() < 1
    assert weights.max() > 1
    assert weights.max() - weights.min() > 1

    # Check that the most common class ("other") has the lowest weight
    assert weights.argmin() == 0


def test_prior_weights_leec12():
    dl = get_dataloader("leec12")
    weights = dl.prior_weights()

    assert weights.min() < 1
    assert weights.max() > 1
    assert weights.max() - weights.min() > 1

    # Check that all frogs have more weight than any bird
    assert weights[6:].min() > weights[:6].max()


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
