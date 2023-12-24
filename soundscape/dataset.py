import os
from glob import glob
from typing import Callable, Union

import numpy as np
import tensorflow as tf
from jax import numpy as jnp
from jax import random
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def _tf2jax(batch):
    """
    Convert a dictionary of tensorflow tensors to a dictionary of jax arrays.
    """

    def _tf2jax_element(element: tf.Tensor | np.ndarray | jnp.ndarray):
        """
        Convert a single tensor to a jax array.
        """

        # Convert the tensor to a numpy array
        if isinstance(element, tf.Tensor):
            element = element.numpy()

        # if the dtype is adequate, convert to a jax array
        if isinstance(element, np.ndarray) and element.dtype != np.dtype("O"):
            element = jnp.array(element)

        return element

    batch = {k: _tf2jax_element(v) for k, v in batch.items()}

    return batch


def read_audio_file(path: str):
    return tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)[0]


def read_image_file(path: str, image_precision: int):
    dtype = tf.uint16 if image_precision == 16 else tf.uint8
    return tf.io.decode_png(tf.io.read_file(path), channels=1, dtype=dtype)


def load_split_metadata(
    rng: random.KeyArray,
    split_dir: str,
    *,
    class_names: list[str],
    class_to_id: Callable[[str], int],
    reading_function: Union[read_audio_file, read_image_file],
):
    """
    Load the metadata for a split of the dataset, including filenames and labels,
    but not the actual image/audio files.

    The data files should be organized in the following way:
        [split_dir]/[class_name]/[file]

    The function class_to_id maps from class names to integer ids. It can be used,
    for instance, to group all bird species into a single class.

    The list class_names should contain a string for each class id, after applying
    the class_to_id mapping.
    """

    filenames = []
    labels = []
    ids = []

    extension = "png" if reading_function == read_image_file else "wav"

    for cls in class_names:
        filenames = glob(os.path.join(split_dir, str(cls), f"*.{extension}"))

        for f in sorted(filenames):
            filenames.append(f)
            labels.append(class_to_id(cls))
            ids.append(int(os.path.basename(f).split(".")[0]))

    # Shuffle the dataset
    rng, _rng = random.split(rng)
    idx = random.permutation(_rng, len(filenames))
    filenames = np.array(filenames)[idx]
    labels = np.array(labels)[idx]
    ids = np.array(ids)[idx]

    return {
        "labels": labels,
        "_files": filenames,
        "_ids": ids,
    }


def get_split_dataloader(
    rng: random.KeyArray,
    split_dir: str,
    *,
    class_names: list[str],
    class_to_id: Callable[[str], int],
    reading_function: Union[read_audio_file, read_image_file],
    cache: bool,
    shuffle_every_epoch: bool,
):
    """
    Get a tf.data.Dataset object for a split of the dataset.
    """

    ds_dict = load_split_metadata(
        rng,
        split_dir,
        class_names=class_names,
        class_to_id=class_to_id,
        reading_function=reading_function,
    )

    ds = tf.data.Dataset.from_tensor_slices(ds_dict).map(
        lambda instance: instance | {"inputs": reading_function(instance["_files"])}
    )

    if cache:
        ds = ds.cache()

    if shuffle_every_epoch:
        ds = ds.shuffle(10000, seed=random.randint(rng))

    return ds


class Dataset:
    def __init__(
        self,
        rng: random.KeyArray,
        *,
        dataset_dir: str,
        reading_function: Union[read_audio_file, read_image_file],
        class_names: list[str],
        class_to_id: Callable[[str], int] = None,
        cache: bool = True,
    ):
        """
        Create a new dataset object, which stores dataloaders for all splits.
        The dataset must be organized in the following way:
            [dataset_dir]/[split_dir]/[class_name]/[file]

        Only the "train" split is shuffled every epoch.
        """

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_to_id = class_to_id.get if class_to_id else class_names.index

        self._splits = {}

        split_names = os.listdir(dataset_dir)
        rngs = random.split(rng, len(split_names))

        for rng, split in zip(rngs, split_names):
            self._splits[split] = get_split_dataloader(
                rng,
                os.path.join(dataset_dir, split),
                class_names=class_names,
                class_to_id=class_to_id,
                reading_function=reading_function,
                cache=cache,
                shuffle=split == "train",  # Only shuffle the training set after caching
            )

    def iterate(
        self,
        rng: random.KeyArray,
        split: str,
        batch_size: int,
        drop_remainder: bool = False,
    ):
        """
        Get an iterator over the dataset for a given split.
        """

        ds = self._splits[split]
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        for batch in ds:
            rng, _rng = random.split(rng)
            _rngs = random.split(_rng, batch_size)
            yield _tf2jax(batch) | {"rng": _rng, "rngs": _rngs}
