import os
from glob import glob

import hydra
import jax
import numpy as np
import tensorflow as tf
from jax import numpy as jnp
from jax import random
from tensorflow.python.ops.numpy_ops import np_config

from ..types import Batch
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


class DataLoader:
    def __init__(
        self,
        rng: jax.Array,
        dataset: Dataset,
        class_order: list[str],
        class_mapping: dict[str, str | None] | None = None,
        cache: bool = True,
        include_test=False,
        batch_size: int = 128,
    ):
        """
        Create a new dataset object, which stores dataloaders for all splits.
        The dataset must be organized in the following way:
            [dataset_dir]/[split_dir]/[class_name]/[file]

        Only the "train" split is shuffled every epoch.
        """

        self.dataset = dataset
        self.class_order = class_order

        if class_mapping is None:
            class_mapping = {cls: cls for cls in self.class_order}

        self.class_mapping = class_mapping
        self.cache = cache
        self.include_test = include_test
        self.batch_size = batch_size

        self._dataloaders = {}

        split_names = os.listdir(dataset.dataset_dir)
        rngs = random.split(rng, len(split_names))

        for rng, split in zip(rngs, split_names):
            if split == "test" and not include_test:
                continue

            rng, rng_metadata, rng_dataloader = random.split(rng, 3)

            split_metadata = self.load_split_metadata(
                rng_metadata, os.path.join(dataset.dataset_dir, split)
            )

            self._dataloaders[split] = self.get_split_dataloader(
                rng_dataloader,
                split_metadata,
                shuffle_every_epoch=split == "train",
                drop_remainder=split == "train",
            )

            if split == "train":
                counts = jnp.bincount(
                    split_metadata["labels"], minlength=len(class_order)
                )
                self.prior = counts / counts.sum()

        self.num_classes = len(class_order)

    def iterate(self, rng: jax.Array, split: str):
        """
        Get an iterator over the dataset for a given split.
        """

        ds = self._dataloaders[split]
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        for batch in ds:
            rng, _rng = random.split(rng)
            _rngs = random.split(_rng, self.batch_size)
            batch = {k: v for k, v in batch.items() if k[0] != "_"}
            yield (Batch(tf2jax(batch) | {"rng": _rng, "rngs": _rngs}))

    def prior_weights(self):
        return 1 / (self.prior * self.num_classes)

    def load_split_metadata(self, rng: jax.Array, split_dir: str):
        """
        Load the metadata for a split of the dataset, including filenames and labels,
        but not the actual image/audio files.

        The data files should be organized in the following way:
            [split_dir]/[class_name]/[file]
        """

        filenames = []
        labels = []
        ids = []

        extension = "wav" if self.dataset.data_type == "audio" else "png"

        existing_classes = os.listdir(split_dir)

        for class_name_original in existing_classes:
            class_name = self.class_mapping.get(class_name_original, None)

            if class_name is None:
                continue

            class_files = glob(
                os.path.join(split_dir, str(class_name_original), f"*.{extension}")
            )

            for f in sorted(class_files):
                filenames.append(f)
                labels.append(self.class_order.index(class_name))
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
        self,
        rng: jax.Array,
        ds_dict: dict,
        shuffle_every_epoch: bool,
        drop_remainder: bool,
    ):
        """
        Get a tf.data.Dataset object for a split of the dataset.
        """

        read_fn = self.dataset.reading_function()

        ds = tf.data.Dataset.from_tensor_slices(ds_dict).map(
            lambda instance: instance | {"inputs": read_fn(instance["_files"])}
        )

        if self.cache:
            ds = ds.cache()

        if shuffle_every_epoch:
            ds = ds.shuffle(100, seed=random.randint(rng, (1,), 0, 2**16)[0])
            # ds = ds.shuffle(10000, seed=random.randint(rng, (1,), 0, 2**16)[0])

        ds = ds.batch(self.batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def __len__(self):
        total = 0
        for dataloader in self._dataloaders.values():
            total += len(dataloader)
        return total

    def get_steps_per_epoch(self, split: str = "train"):
        return len(self._dataloaders[split])


def get_dataloader(rng, dataloader_settings, **kwargs) -> DataLoader:
    return hydra.utils.instantiate(dataloader_settings, rng=rng, **kwargs)
