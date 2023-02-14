from soundscape import dataset, settings

import os
import jax
from jax import random, numpy as jnp
import numpy as np
import tensorflow as tf
import pytest
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_read_audio_file():
    filename = "test/data/0.wav"
    audio = dataset.read_audio_file(filename)
    assert audio.shape == (22050 * 5, 1)


def test_read_image_file():
    filename = "test/data/0.png"
    img = dataset.read_image_file(filename, precision=16)
    assert img.shape == (256, 423, 1)
    assert img.dtype == tf.uint16


def test_get_classes():
    class_order = ["other", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    class_order_2 = ["first_6", "last_6"]

    settings.from_dict(
        {"class_order": class_order, "class_order_2": class_order_2, "num_classes": 12}
    )

    classes, idx_fn = dataset.get_classes(
        num_classes=13,
    )
    class_names = dataset.get_class_names(num_classes=13)

    assert classes == class_order == class_names
    assert idx_fn("a") == 1
    assert idx_fn("other") == 0
    assert idx_fn("l") == 12

    classes, idx_fn = dataset.get_classes(
        num_classes=12,
    )
    class_names = dataset.get_class_names(num_classes=12)

    assert classes == class_order[1:] == class_names
    assert idx_fn("a") == 0
    assert idx_fn("l") == 11
    assert "other" not in classes

    classes, idx_fn = dataset.get_classes(
        num_classes=2,
    )
    class_names = dataset.get_class_names(num_classes=2)

    assert classes == class_order_2 == class_names
    assert idx_fn("a") == 0
    assert idx_fn("b") == 0
    assert idx_fn("f") == 0
    assert idx_fn("g") == 1
    assert idx_fn("i") == 1
    assert idx_fn("l") == 1
    assert "other" not in classes

    with pytest.raises(ValueError):
        dataset.get_classes(num_classes=11)


@pytest.mark.parametrize(
    "num_classes, num_train, num_val, num_test",
    [(13, 9990, 3323, 3323), (12, 2293, 758, 758), (2, 2293, 758, 758)],
)
def test_dataset(num_classes, num_train, num_val, num_test):
    settings.from_dict(
        {
            "data_dir": "data",
            "spectrogram_dir": "leec",
            "num_classes": 13,
            "class_order": [
                "other",
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
            ],
            "class_order_2": ["bird", "frog"],
            "extension": "png",
        }
    )

    class_names = dataset.get_class_names(num_classes=num_classes)

    rng = random.PRNGKey(0)

    train_ds = dataset.get_dataset_dict("train", rng, num_classes=num_classes)
    assert len(train_ds["_file"]) == num_train
    assert len(train_ds["labels"]) == num_train
    assert len(train_ds["id"]) == num_train

    val_ds = dataset.get_dataset_dict("val", rng, num_classes=num_classes)
    assert len(val_ds["_file"]) == num_val
    assert len(val_ds["labels"]) == num_val
    assert len(val_ds["id"]) == num_val

    test_ds = dataset.get_dataset_dict("test", rng, num_classes=num_classes)
    assert len(test_ds["_file"]) == num_test
    assert len(test_ds["labels"]) == num_test
    assert len(test_ds["id"]) == num_test

    assert set(train_ds["id"]) & set(val_ds["id"]) == set()
    assert set(train_ds["id"]) & set(test_ds["id"]) == set()
    assert set(val_ds["id"]) & set(test_ds["id"]) == set()

    assert set(train_ds["_file"]) & set(val_ds["_file"]) == set()
    assert set(train_ds["_file"]) & set(test_ds["_file"]) == set()
    assert set(val_ds["_file"]) & set(test_ds["_file"]) == set()

    assert len(set(train_ds["id"])) == num_train
    assert len(set(val_ds["id"])) == num_val
    assert len(set(test_ds["id"])) == num_test

    assert (
        set(train_ds["labels"])
        == set(val_ds["labels"])
        == set(test_ds["labels"])
        == set(range(num_classes))
    )

    mean_train = jax.nn.one_hot(train_ds["labels"], 13).mean(0)
    mean_val = jax.nn.one_hot(val_ds["labels"], 13).mean(0)
    mean_test = jax.nn.one_hot(test_ds["labels"], 13).mean(0)

    assert jnp.allclose(mean_train, mean_val, atol=0.005)
    assert jnp.allclose(mean_train, mean_test, atol=0.005)
    assert jnp.allclose(mean_val, mean_test, atol=0.005)

    if num_classes in [13, 12]:
        for ds, split_name in zip(
            (train_ds, val_ds, test_ds), ("train", "val", "test")
        ):
            for i in range(10):
                class_id = ds["labels"][i]
                class_name = class_names[class_id]
                idx = ds["id"][i]
                files = os.listdir(f"data/leec/{split_name}/{class_name}/")

                assert f"{idx}.png" in files
                assert f"{idx}.wav" in files


def test_get_tensorflow_dataset():
    settings.from_dict(
        {
            "data_dir": "data",
            "spectrogram_dir": "leec",
            "num_classes": 13,
            "class_order": [
                "other",
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
            ],
            "class_order_2": ["bird", "frog"],
            "extension": "png",
            "precision": 16,
        }
    )

    rng = jax.random.PRNGKey(0)

    train_ds = dataset.get_tensorflow_dataset("train", rng)
    val_ds = dataset.get_tensorflow_dataset("val", rng)
    test_ds = dataset.get_tensorflow_dataset("test", rng)

    train_dict = dataset.get_dataset_dict("train", rng)
    val_dict = dataset.get_dataset_dict("val", rng)
    test_dict = dataset.get_dataset_dict("test", rng)

    for ds, ds_dict in zip(
        (train_ds, val_ds, test_ds), (train_dict, val_dict, test_dict)
    ):
        for i, batch in enumerate(ds.take(32)):
            assert batch["inputs"].shape == (256, 423, 1)
            assert batch["labels"] == ds_dict["labels"][i]
            assert batch["id"] == ds_dict["id"][i]
            assert batch["_file"] == ds_dict["_file"][i]

    train_ds_audio = dataset.get_tensorflow_dataset("train", rng, extension="wav")
    val_ds_audio = dataset.get_tensorflow_dataset("val", rng, extension="wav")
    test_ds_audio = dataset.get_tensorflow_dataset("test", rng, extension="wav")

    train_dict = dataset.get_dataset_dict("train", rng, extension="wav")
    val_dict = dataset.get_dataset_dict("val", rng, extension="wav")
    test_dict = dataset.get_dataset_dict("test", rng, extension="wav")

    for ds, ds_dict in zip(
        (train_ds_audio, val_ds_audio, test_ds_audio), (train_dict, val_dict, test_dict)
    ):
        for i, batch in enumerate(ds.take(32)):
            assert batch["inputs"].shape == (22050 * 5, 1)
            assert batch["labels"] == ds_dict["labels"][i]
            assert batch["id"] == ds_dict["id"][i]
            assert batch["_file"] == ds_dict["_file"][i]


def test_tf2jax():

    a = {
        "inputs": tf.random.uniform((10, 256, 256, 3)),
        "labels": tf.random.uniform((10,), minval=0, maxval=13, dtype=tf.int32),
        "id": tf.random.uniform((10,), minval=0, maxval=1000, dtype=tf.int32),
        "rngs": tf.random.uniform((10, 2), minval=0, maxval=1000, dtype=tf.int32),
        "_file": tf.constant(["string"] * 10),
    }

    b = dataset.tf2jax(a)

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


def test_split_rng():
    rng1 = jax.random.PRNGKey(0)
    rng2 = jax.random.PRNGKey(1)

    values1 = {"rng": rng1, "inputs": jnp.zeros((10, 10))}
    values2 = {"rng": rng2, "inputs": jnp.ones((12, 10))}

    values1 = dataset.split_rng(values1)
    values2 = dataset.split_rng(values2)

    assert not jnp.array_equal(values1["rng"], values2["rng"])
    assert not jnp.array_equal(values1["rng"], rng1)
    assert not jnp.array_equal(values2["rng"], rng2)
    assert values1["rngs"].shape == (10, 2)
    assert values2["rngs"].shape == (12, 2)


def test_prepare_image():

    rng = jax.random.PRNGKey(0)
    img = jax.random.randint(rng, (10, 1000, 128, 1), 0, 2**16 - 1, dtype=jnp.uint16)

    img2 = dataset.prepare_image({"inputs": img}, precision=16)["inputs"]

    assert img2.shape == (10, 1000, 128, 3)
    assert img2.dtype == jnp.float32
    assert img2.min() >= 0
    assert img2.max() <= 1

    assert jnp.allclose(img2.mean(), 0.5, atol=0.01)
    assert jnp.allclose(img2[..., 0], img2[..., 1], atol=0.01)
    assert jnp.allclose(img2[..., 0], img2[..., 2], atol=0.01)
    assert jnp.allclose(img2[..., 0], img[..., 0] / 2**16, atol=0.01)


def test_downsample_image():
    rng = jax.random.PRNGKey(0)
    img = jax.random.randint(rng, (10, 1000, 128, 2), 0, 2**16 - 1, dtype=jnp.uint16)

    img2 = dataset.downsample_image({"inputs": img})["inputs"]

    assert img2.shape == (10, 224, 224, 2)
    assert jnp.linalg.matrix_rank(img2.reshape(10, -1)) == 10


def test_one_hot_encode():
    labels = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]
    )
    new_labels = dataset.one_hot_encode({"labels": labels}, num_classes=10)

    assert jnp.allclose(new_labels["one_hot_labels"][0], jnp.eye(10))
    assert jnp.allclose(new_labels["one_hot_labels"][1], jnp.eye(10)[::-1])
    assert jnp.allclose(new_labels["labels"], labels)


@pytest.mark.parametrize("num_classes", [2, 12, 13])
def test_get_class_weights(num_classes):
    settings.from_dict(
        {
            "data_dir": "data",
            "labels_file": "labels.csv",
            "num_classes": 13,
            "class_order": [
                "other",
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
            ],
            "class_order_2": ["bird", "frog"],
        }
    )
    weights = dataset.get_class_weights(num_classes=num_classes)
    classes, class_idx_fn = dataset.get_classes(num_classes=num_classes)
    assert len(weights) == num_classes

    df = pd.read_csv("data/labels.csv")
    df = df[df["exists"]]

    if num_classes != 13:
        df = df[df["class"] != "other"]

    class_ids = df["class"].map(class_idx_fn)

    class_freqs = class_ids.value_counts(normalize=True).sort_index().to_numpy()

    assert jnp.allclose(class_freqs * weights, 1 / num_classes)
