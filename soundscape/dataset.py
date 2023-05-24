import tensorflow as tf
import pickle
import jax
import pandas as pd
import numpy as np
from jax import random, numpy as jnp
import os
from glob import glob
import scipy.special as sc

from .settings import settings_fn
from .composition import Composable


def read_audio_file(path):
    """
    Read an audio file.
    """

    data = tf.io.read_file(path)
    return tf.audio.decode_wav(data, desired_channels=1)[0]


@settings_fn
def read_image_file(path, *, precision):
    """
    Read an image file.

    Settings:
    ---------
    precision: int
        The precision of the image. Can be 8 or 16.
    """
    data = tf.io.read_file(path)
    dtype = tf.uint16 if precision == 16 else tf.uint8
    return tf.io.decode_png(data, channels=1, dtype=dtype)


@settings_fn
def get_classes(*, class_order, class_order_2, num_classes):
    """
    Get a list of class names and a function that maps a class name to a class id.

    Settings:
    ---------
    class_order: list
        The order of the 13 classes ("other" + 6 birds + 6 frogs)
    class_order_2: list
        The order of the 2 classes ("bird" and "frog")
    num_classes: int
        The number of classes to use. Can be 13, 12 or 2.
    """

    # Create a copy of the class order list
    classes = [c for c in class_order]

    # If using 13 classes, return the classes list as is
    if num_classes == 13:
        return classes, classes.index

    # If using 12 classes, remove the "other" class before returning
    if num_classes == 12:
        classes.remove("other")
        return classes, classes.index

    # If using 2 classes, return the 2 classes list and a function that maps
    # each species' name to either 0 or 1
    if num_classes == 2:
        classes.remove("other")
        return class_order_2, lambda x: int(classes.index(x) >= 6)

    return classes, classes.index


@settings_fn
def get_class_names():
    """
    Get a list of class names
    """

    return get_classes()[0]


@settings_fn
def get_dataset_dict(
    split, rng, *, data_dir, spectrogram_dir, num_classes, class_order, extension
):
    """
    Return a dictionary containing the dataset entries.

    Parameters:
    ----------
    split: str
        The split to use. Can be "train", "val" or "test".
    rng: jax.random.PRNGKey
        The random number generator key used to shuffle the dataset.

    Settings:
    ---------
    data_dir: str
        The path to the data directory.
    spectrogram_dir: str
        The name of the directory containing the spectrograms.
    num_classes: int
        The number of classes to use. Can be 13, 12 or 2.
    class_order: list
        The order of the 13 classes ("other" + 6 birds + 6 frogs)
    extension: str
        The extension of the spectrograms. Can be "png" or "wav".

    """

    # Get the path to the dataset directory
    dataset_dir = os.path.join(data_dir, spectrogram_dir, split)

    # Get the list of classes
    classes = [c for c in class_order]
    if num_classes != 13 and "other" in classes:
        classes.remove("other")

    # Get the function that maps a class name to a class id
    class_to_id = get_classes()[1]

    # Create the dataset dictionary
    ds = {
        "_file": [],
        "labels": [],
        "id": [],
    }

    for cls in classes:
        # Get the list of files in the current class directory
        files = glob(os.path.join(dataset_dir, str(cls), f"*.{extension}"))

        # Add the files to the dataset dictionary
        for f in sorted(files):
            ds["_file"].append(f)
            ds["labels"].append(class_to_id(cls))
            ds["id"].append(int(os.path.basename(f).split(".")[0]))

    # Shuffle the dataset
    idx = random.permutation(rng, len(ds["_file"]))
    ds = {k: np.array(v)[idx] for k, v in ds.items()}

    return ds


@settings_fn
def get_tensorflow_dataset(split, rng, *, extension):
    """
    Return a tf.data.Dataset object containing the dataset entries.

    Parameters:
    ----------
    split: str
        The split to use. Can be "train", "val" or "test".
    rng: jax.random.PRNGKey
        The random number generator key used to shuffle the dataset.

    Settings:
    ---------
    extension: str
        The extension of the spectrograms. Can be "png" or "wav".
    """

    # Get the dataset dictionary
    ds_dict = get_dataset_dict(split, rng)

    # Create the tensorflow dataset
    ds = tf.data.Dataset.from_tensor_slices(ds_dict)

    # Map a function that reads the data from the files
    read_fn = read_image_file if extension == "png" else read_audio_file
    ds = ds.map(lambda x: {"inputs": read_fn(x["_file"]), **x})

    return ds


@Composable
def tf2jax(values):
    """
    Convert a dictionary of tensorflow tensors to a dictionary of jax arrays.
    """

    def _tf2jax(x):
        """
        Convert a single tensor to a jax array.
        """

        # Convert the tensor to a numpy array
        if isinstance(x, tf.Tensor):
            x = x.numpy()

        # if the dtype is adequate, convert to a jax array
        if isinstance(x, np.ndarray) and x.dtype != np.dtype("O"):
            x = jnp.array(x)

        return x

    values = {k: _tf2jax(v) for k, v in values.items()}

    return values


@settings_fn
def get_class_weights(*, data_dir, labels_file, num_classes):
    """
    Return an array with weights for each class.
    These weights can be used to create metrics that assign an
    equal importance to each class, regardless of its frequency.

    Settings:
    ---------
    data_dir: str
        The path to the data directory.
    labels_file: str
        The name of the file containing the labels.
    num_classes: int
        The number of classes to use. Can be 13, 12 or 2.
    """

    # Read the labels file
    df = pd.read_csv(os.path.join(data_dir, labels_file))

    # Filters only the relevant samples
    df = df[df["exists"]]
    if num_classes != 13:
        df = df[df["selected"]]

    # Maps the class names to ids
    classes, class_to_id = get_classes()
    df_classes = df["class"].map(class_to_id)

    # Compute the frequency of each class
    frequencies = df_classes.value_counts(True)
    frequencies = frequencies.sort_index().to_numpy()

    # Compute the weights
    weights = 1 / (frequencies * num_classes)

    return jnp.array(weights)


@Composable
def split_rng(values):
    """
    Split the rng key into one key for each element in the batch.
    """

    rng = values["rng"]
    inputs = values["inputs"]

    rng, _rng = jax.random.split(rng, 2)
    rngs = jax.random.split(_rng, inputs.shape[0])

    return {**values, "rng": rng, "rngs": rngs}


@Composable
@settings_fn
def prepare_image(values, *, precision):
    """
    Normalize a repeat an image's channels 3 times.

    Settings:
    ---------
    precision: int
        The number of bits used to represent each pixel.
    """

    image = values["inputs"]

    # Normalize the image
    image = jnp.float32(image) / (2**precision)

    # Repeat the channels 3 times
    image = jnp.repeat(image, 3, axis=-1)

    return {**values, "inputs": image}


@Composable
def downsample_image(values):
    """
    Downsample an image to 224x224 pixels.
    """

    image = values["inputs"]

    shape = (image.shape[0], 224, 224, image.shape[-1])
    image = jax.image.resize(image, shape, method="bicubic")

    return {**values, "inputs": image}


@Composable
@settings_fn
def one_hot_encode(values, *, num_classes):
    """
    Convert a class name to a one-hot encoded vector

    Settings:
    ---------
    num_classes: int
        The number of classes to use. Can be 13, 12 or 2.
    """

    labels = values["labels"]

    labels = jax.nn.one_hot(labels, num_classes)

    return {**values, "one_hot_labels": labels}


"""
Tensorflow versions of the functions defined above.
"""


@Composable
@settings_fn
def prepare_image_tf(values, *, precision):
    image = values["inputs"]
    image = tf.cast(image, tf.float32) * (2 ** (-precision))
    image = tf.repeat(image, 3, axis=-1)
    return {**values, "inputs": image}


@Composable
def downsample_image_tf(values):
    image = values["inputs"]
    shape = (image.shape[0], 224, 224, image.shape[-1])
    image = tf.image.resize(image, shape[1:3], method="bicubic")
    return {**values, "inputs": image}


@Composable
def one_hot_encode_tf(values):
    """
    Convert a class name to a one-hot encoded vector
    """

    classes, class_to_id = get_classes()

    labels = values["labels"]

    labels = tf.one_hot(labels, len(classes))

    return {**values, "one_hot_labels": labels}
