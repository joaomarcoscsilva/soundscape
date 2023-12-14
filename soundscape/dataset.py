import os
from glob import glob

import numpy as np
import tensorflow as tf
from jax import random
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def read_audio_file(path):
    return tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)[0]


def read_image_file(path, image_precision):
    dtype = tf.uint16 if image_precision == 16 else tf.uint8
    return tf.io.decode_png(tf.io.read_file(path), channels=1, dtype=dtype)


def load_dataset_split(rng, split_dir, *, class_names, class_to_id, reading_function):
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


def get_dataloader(
    rng,
    split_dir,
    *,
    class_names,
    class_to_id,
    reading_function,
    cache,
    batch_size,
    shuffle,
    drop_remainder=False,
):
    ds_dict = load_dataset_split(
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

    if shuffle:
        ds = ds.shuffle(10000, seed=random.randint(rng))

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


class Dataset:
    def __init__(
        self,
        rng,
        *,
        dataset_dir,
        batch_size,
        reading_function,
        class_names,
        class_to_id,
    ):
        self.class_names = class_names
        self.class_to_id = class_to_id.get if class_to_id else class_names.index

        rng, rng_train, rng_val, rng_test = random.split(rng, 4)

        self.train = get_dataloader(
            rng_train,
            os.path.join(dataset_dir, "train"),
            class_names=class_names,
            class_to_id=class_to_id,
            reading_function=reading_function,
            cache=True,
            batch_size=batch_size,
            shuffle=True,
            drop_remainder=False,
        )

        self.val = get_dataloader(
            rng_val,
            os.path.join(dataset_dir, "val"),
            class_names=class_names,
            class_to_id=class_to_id,
            reading_function=reading_function,
            cache=True,
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )

        self.test = get_dataloader(
            rng_test,
            os.path.join(dataset_dir, "test"),
            class_names=class_names,
            class_to_id=class_to_id,
            reading_function=reading_function,
            cache=True,
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
