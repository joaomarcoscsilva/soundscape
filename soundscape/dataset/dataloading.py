import os
from glob import glob
from typing import Callable

import jax
import numpy as np
import tensorflow as tf
from jax import numpy as jnp
from jax import random
from tensorflow.python.ops.numpy_ops import np_config

from .dataset import Dataset

np_config.enable_numpy_behavior()


def tf2jax(batch):
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


def load_split_metadata(
    rng: jax.Array,
    split_dir: str,
    dataset: Dataset,
    class_to_id: Callable[[str], int],
    skipped_classes: list[str] = [],
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

    extension = "wav" if dataset.data_type == "audio" else "png"

    for cls in dataset.class_order:
        if cls in skipped_classes:
            continue

        class_files = glob(os.path.join(split_dir, str(cls), f"*.{extension}"))

        for f in sorted(class_files):
            filenames.append(f)
            labels.append(class_to_id(cls))
            ids.append(int(os.path.basename(f).split(".")[0]))

    # Shuffle the dataset
    rng, _rng = random.split(rng)
    idx = random.permutation(_rng, len(filenames))
    filenames = np.array(filenames)[idx]
    labels = np.array(labels)[idx]
    ids = np.array(ids)[idx]

    num_classes = len(set(labels))

    return {
        "labels": labels,
        "_files": filenames,
        "_ids": ids,
    }


def get_split_dataloader(
    rng: jax.Array,
    ds_dict: dict,
    dataset: Dataset,
    cache: bool,
    shuffle_every_epoch: bool,
):
    """
    Get a tf.data.Dataset object for a split of the dataset.
    """

    read_fn = dataset._reading_function()

    ds = tf.data.Dataset.from_tensor_slices(ds_dict).map(
        lambda instance: instance | {"inputs": read_fn(instance["_files"])}
    )

    if cache:
        ds = ds.cache()

    if shuffle_every_epoch:
        ds = ds.shuffle(10000, seed=random.randint(rng, (1,), 0, 2**16)[0])

    return ds


class DataLoader:
    def __init__(
        self,
        rng: jax.Array,
        dataset: Dataset,
        data_type: str,
        class_to_id: Callable[[str], int] = None,
        cache: bool = True,
        skipped_classes: list[str] = [],
    ):
        """
        Create a new dataset object, which stores dataloaders for all splits.
        The dataset must be organized in the following way:
            [dataset_dir]/[split_dir]/[class_name]/[file]

        Only the "train" split is shuffled every epoch.
        """

        self.dataset = dataset
        self.class_to_id = class_to_id if class_to_id else dataset.class_order.index
        self.data_type = data_type

        self._dataloaders = {}

        split_names = os.listdir(dataset.dataset_dir)
        rngs = random.split(rng, len(split_names))

        possible_labels = set()

        for rng, split in zip(rngs, split_names):
            rng, rng_metadata, rng_dataloader = random.split(rng, 3)

            split_metadata = load_split_metadata(
                rng_metadata,
                os.path.join(dataset.dataset_dir, split),
                dataset,
                class_to_id,
                skipped_classes=skipped_classes,
            )

            self._dataloaders[split] = get_split_dataloader(
                rng_dataloader,
                split_metadata,
                dataset,
                cache=cache,
                shuffle_every_epoch=split == "train",
            )

            possible_labels.update(split_metadata["labels"])

        self.num_classes = len(possible_labels)

    def iterate(
        self,
        rng: jax.Array,
        split: str,
        batch_size: int,
        drop_remainder: bool = False,
    ):
        """
        Get an iterator over the dataset for a given split.
        """

        ds = self._dataloaders[split]
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        for batch in ds:
            rng, _rng = random.split(rng)
            _rngs = random.split(_rng, batch_size)
            yield tf2jax(batch) | {"rng": _rng, "rngs": _rngs}
