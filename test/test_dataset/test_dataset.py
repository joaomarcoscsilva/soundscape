import os

import tensorflow as tf
from omegaconf import DictConfig

from soundscape.dataset import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_read_audio_file():
    filename = "test/test_dataset/data/0.wav"
    audio = dataset._read_audio_file(filename)
    assert audio.shape == (22050 * 5, 1)


def test_read_image_file():
    filename = "test/test_dataset/data/0.png"
    img = dataset._read_image_file(filename, image_precision=16)
    assert img.shape == (256, 423, 1)
    assert img.dtype == tf.uint16


def test_labels_dataframe():
    df = dataset.load_labels_dataframe("data/labels.csv")

    assert df["exists"].all()
    assert df.shape == (16636, 12)
    assert (df["begin_time"] >= 0.0).all()
    assert df["class"].unique().shape == (13,)


def test_dataset_reading_fn():
    filename = "test/test_dataset/data/0.wav"
    ds = dataset.Dataset(data_type="audio")
    wav = ds.reading_function()(filename)
    assert tf.reduce_all(wav == dataset._read_audio_file(filename))

    filename = "test/test_dataset/data/0.png"
    ds = dataset.Dataset(
        data_type="image",
        preprocessing=DictConfig({"image_precision": 16}),
    )
    img = ds.reading_function()(filename)
    assert tf.reduce_all(img == dataset._read_image_file(filename, 16))
