import tensorflow as tf
import pandas as pd
import numpy as np
from jax import random, numpy as jnp
import os
from glob import glob

from . import settings
from .composition import Composable

settings_fn, settings_dict = settings.from_file("dataset_settings.yaml")


@settings_fn
def read_audio_file_tf(path):
    data = tf.io.read_file(path)
    return tf.audio.decode_wav(data, desired_channels=1)[0]


@settings_fn
def read_image_file_tf(path, *, precision):
    data = tf.io.read_file(path)
    dtype = tf.uint16 if precision == 16 else tf.uint8
    return tf.io.decode_png(data, channels=1, dtype=dtype)


@settings_fn
def get_classes(*, class_order, class_order_2, num_classes):
    """
    Return a list with class names and a function that maps a class name to its id
    """

    classes = [c for c in class_order]

    if num_classes == 13:
        return classes, classes.index

    if num_classes == 12:
        classes.remove("other")
        return classes, classes.index

    if num_classes == 2:
        classes.remove("other")
        return class_order_2, lambda x: int(classes.index(x) >= 6)

    raise ValueError("Invalid number of classes")


@settings_fn
def get_class_names():
    return get_classes()[0]


@settings_fn
def get_dataset(
    split, rng, *, data_dir, spectrogram_dir, num_classes, class_order, extension
):
    """
    Return a tf.data.Dataset object
    """

    dataset_dir = os.path.join(data_dir, spectrogram_dir, split)

    classes = [c for c in class_order]
    if num_classes != 13:
        classes.remove("other")

    class_to_id = get_classes()[1]

    ds = {
        "file": [],
        "labels": [],
        "id": [],
        "rng": [],
    }

    for cls in classes:
        files = glob(os.path.join(dataset_dir, cls, f"*.{extension}"))
        for f in sorted(files):
            ds["file"].append(f)
            ds["labels"].append(class_to_id(cls))
            ds["id"].append(int(os.path.basename(f).split(".")[0]))

            rng, _rng = random.split(rng)
            ds["rng"].append(_rng)

    idx = random.permutation(rng, len(ds["file"]))
    ds = {k: np.array(v)[idx] for k, v in ds.items()}

    return ds


@settings_fn
def get_tensorflow_dataset(split, rng, *, extension):
    ds_dict = get_dataset(split, rng)
    ds = tf.data.Dataset.from_tensor_slices(ds_dict)
    read_fn = read_image_file_tf if extension == "png" else read_audio_file_tf
    ds = ds.map(lambda x: {"inputs": read_fn(x["file"]), **x})
    ds = ds.cache()
    return ds


@Composable
def tf2jax(values):
    """
    Convert a tf.data.Dataset object to a dictionary of numpy arrays
    """

    for k, v in values.items():
        if isinstance(v, tf.Tensor):
            values[k] = v.numpy()
        if not (isinstance(values[k], bytes) or values[k].dtype == np.dtype("O")):
            values[k] = jnp.array(values[k])

    return values


@Composable
@settings_fn
def prepare_image(values, *, precision):
    image = values["inputs"]
    image = tf.cast(image, tf.float32) / (2**precision)
    image = tf.repeat(image, 3, axis=-1)
    return {**values, "inputs": image}


@Composable
def downsample_image(values):
    image = values["inputs"]
    shape = (image.shape[0], 224, 224, image.shape[-1])
    # image = jax.image.resize(image, shape, method="bicubic")
    image = tf.image.resize(image, shape[1:3], method="bicubic")
    return {**values, "inputs": image}


@Composable
@settings_fn
def prepare_image_channels(values, *, channel_mean, channel_std):
    image = values["inputs"]
    # image = image - jnp.array(channel_mean)[None, None, None, :]
    # image = image / jnp.array(channel_std)[None, None, None, :]
    image = image - tf.constant(channel_mean)[None, None, None, :]
    image = image / tf.constant(channel_std)[None, None, None, :]
    return {**values, "inputs": image}


@Composable
def one_hot_encode(values):
    """
    Convert a class name to a one-hot encoded vector
    """

    class_names = get_class_names()

    labels = values["labels"]
    # labels = jnp.array([class_to_id(l.decode()) for l in labels])
    # labels = jax.nn.one_hot(labels, len(class_names))

    labels = tf.one_hot(labels, len(class_names))

    return {**values, "labels": labels}


@settings_fn
def get_class_weights(*, data_dir, labels_file, num_classes):
    df = pd.read_csv(os.path.join(data_dir, labels_file))

    df = df[df["exists"]]
    if num_classes != 13:
        df = df[df["selected"]]

    classes, class_to_id = get_classes()
    df_classes = df["class"].map(class_to_id)
    weights = df_classes.value_counts(True)
    weights = weights.sort_index()

    weights = weights.to_numpy()

    weights = 1 / (weights * num_classes)

    return weights
