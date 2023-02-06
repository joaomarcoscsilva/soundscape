from soundscape import dataset, settings

import os
import jax
from jax import random, numpy as jnp
import tensorflow as tf
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_read_audio_file():
    filename = "test/data/audio.wav"
    audio = dataset.read_audio_file(filename)
    assert audio.shape == (22050 * 5, 1)


def test_read_image_file():
    filename = "test/data/img.png"
    img = dataset.read_image_file(filename, precision=16)
    assert img.shape == (10328, 256, 1)
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

    train_ds = dataset.get_dataset("train", rng, num_classes=num_classes)
    assert len(train_ds["_file"]) == num_train
    assert len(train_ds["labels"]) == num_train
    assert len(train_ds["id"]) == num_train
    assert len(train_ds["rngs"]) == num_train

    val_ds = dataset.get_dataset("val", rng, num_classes=num_classes)
    assert len(val_ds["_file"]) == num_val
    assert len(val_ds["labels"]) == num_val
    assert len(val_ds["id"]) == num_val
    assert len(val_ds["rngs"]) == num_val

    test_ds = dataset.get_dataset("test", rng, num_classes=num_classes)
    assert len(test_ds["_file"]) == num_test
    assert len(test_ds["labels"]) == num_test
    assert len(test_ds["id"]) == num_test
    assert len(test_ds["rngs"]) == num_test

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
