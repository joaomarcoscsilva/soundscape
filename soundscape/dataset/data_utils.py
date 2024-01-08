import jax
from jax import numpy as jnp


def prepare_images(batch):
    """
    Normalize a repeat an image's channels 3 times.
    """

    images = batch["inputs"]
    image_precision = images.dtype.itemsize * 8
    images = jnp.float32(images) / (2**image_precision)
    images = jnp.repeat(images, 3, axis=-1)

    return batch | {"inputs": images}


def one_hot_encode(batch, num_classes):
    """
    Convert a class name to a one-hot encoded vector
    """

    labels = batch["labels"]
    label_probs = jax.nn.one_hot(labels, num_classes)

    return batch | {"label_probs": label_probs}


def downsample_images(batch):
    """
    Downsample an image to 224x224 pixels.
    """

    images = batch["inputs"]
    shape = (*images.shape[:-3], 224, 224, images.shape[-1])
    images = jax.image.resize(images, shape, method="bicubic")

    return batch | {"inputs": images}
