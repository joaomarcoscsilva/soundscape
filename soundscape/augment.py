from functools import partial
from typing import Protocol, Union

import jax
from jax import numpy as jnp
from jax import random

from .composition import Composable, identity
from .settings import settings_fn

"""
Define vmapped versions of some functions in jax.random.

These are used because each sample in a batch has its own random key.
"""

batch_split = jax.vmap(lambda rng, n: tuple(random.split(rng, n)), in_axes=(0, None))

batch_uniform = jax.vmap(
    lambda rng, minval, maxval: random.uniform(rng, minval=minval, maxval=maxval),
    in_axes=(0, None, None),
)

batch_beta = jax.vmap(
    lambda rng, a, b: random.beta(rng, a=a, b=b),
    in_axes=(0, None, None),
)


def crop_time_array(array, begin_time, *, segment_length, cropped_length, axis):
    """
    Crop an array along the time axis.

    Parameters:
    -----------
    array: jnp.ndarray
        The array to crop.
    begin_time: float
        The time (in seconds) to begin the crop.
    segment_length: float
        The duration (in seconds) that the original array represents.
    cropped_length: float
        The duration (in seconds) of the cropped array.
    axis: int
        The axis that represents the time dimension.

    Returns:
    --------
    jnp.ndarray
        The cropped array.
    """

    duration_idx = int((cropped_length / segment_length) * array.shape[axis])
    durations = array.shape[:axis] + (duration_idx,) + array.shape[axis + 1 :]

    begin_idx = jnp.int32(begin_time * array.shape[axis] / segment_length)

    begin_indices = [0] * len(array.shape)
    begin_indices[axis] = begin_idx

    return jax.lax.dynamic_slice(array, begin_indices, durations)


@settings_fn
def deterministic_time_crop(*, segment_length, cropped_length, extension):
    """
    Crop a centered segment in a time array.

    Parameters:
    -----------
    values["inputs"]: jnp.ndarray
        A batch of arrays to crop.

    Settings:
    ---------
    segment_length: float
        The duration (in seconds) that the original array represents.
    cropped_length: float
        The duration (in seconds) of the cropped array.
    extension: str
        The extension of the data type. This is used to determine the time axis.

    Returns:
    --------
    values["inputs"]: jnp.ndarray
        The cropped arrays.
    """

    # Define the vmapped version of crop_time_array
    _batch_crop_time_array = jax.vmap(
        partial(
            crop_time_array,
            segment_length=segment_length,
            cropped_length=cropped_length,
            axis=1 if extension == "png" else 0,
        )
    )

    # Find the beginning time for the crops.
    begin_time = (segment_length - cropped_length) / 2

    def _deterministic_time_crop(values):
        begin_times = jnp.repeat(begin_time, len(values["inputs"]))

        # Crop the arrays
        new_inputs = _batch_crop_time_array(values["inputs"], begin_times)

        return {**values, "inputs": new_inputs}

    return _deterministic_time_crop


@settings_fn
def random_time_crop(*, segment_length, cropped_length, extension):
    """
    Crop a random segment in a time array.

    Parameters:
    -----------
    values["inputs"]: jnp.ndarray
        A batch of arrays to crop.
    values["rngs"]: jnp.ndarray
        A batch of random keys.

    Settings:
    ---------
    segment_length: float
        The duration (in seconds) that the original array represents.
    cropped_length: float
        The duration (in seconds) of the cropped array.
    extension: str
        The extension of the data type. This is used to determine the time axis.

    Returns:
    --------
    values["inputs"]: jnp.ndarray
        The cropped arrays.
    values["rngs"]: jnp.ndarray
        The new random keys.
    """

    # Define the vmapped version of crop_time_array
    _batch_crop_time_array = jax.vmap(
        partial(
            crop_time_array,
            segment_length=segment_length,
            cropped_length=cropped_length,
            axis=1 if extension == "png" else 0,
        )
    )

    def _random_time_crop(values):
        # Batch split the random keys
        rngs, _rngs = batch_split(values["rngs"], 2)

        # Generate random begin times
        begin_times = batch_uniform(_rngs, 0, segment_length - cropped_length)

        # Crop the arrays
        new_inputs = _batch_crop_time_array(values["inputs"], begin_times)

        return {**values, "inputs": new_inputs, "rngs": rngs}

    return _random_time_crop


@settings_fn
def time_crop(*, crop_type="deterministic"):
    """
    Return a composable time cropping function.

    Settings:
    ---------
    crop_type: str
        The type of time cropping to use. Can be "deterministic", "random" or "none".

    Returns:
    --------
    Composable
        A composable time cropping function.
    """

    if crop_type == "deterministic":
        return deterministic_time_crop()
    elif crop_type == "random":
        return random_time_crop()
    else:
        return identity


@partial(jax.vmap, in_axes=(0, None, 0))
def rectangular_mask(rng, image_shape, ratio):
    """
    Generate a mask matrix full of 1s, except for a rectangular region of 0s.
    The fraction of the image that is masked is roughly equal to the ratio parameter.

    Parameters:
    -----------
    rng: jnp.ndarray
        A random key.
    image_shape: tuple
        The shape of the image to mask.
    ratio: float
        The fraction of the image to mask.

    Returns:
    --------
    jnp.ndarray
        The mask matrix.
    jnp.ndarray
        The ratio of the masked region to the total image area.
        It should be close to the ratio parameter.
    """

    # Split the random key
    rng, rng_row, rng_col = jax.random.split(rng, 3)

    # Find the shape of the masked region using the given ratio
    mask_shape = jnp.int32(jnp.array(image_shape) * jnp.sqrt(1 - ratio))

    # Generate coordinates for one corner of the masked region
    # The coordinates are generated such that the masked region is always
    # contained inside the image.
    row_begin = jax.random.randint(rng_row, (1,), 0, image_shape[0] - mask_shape[0])
    col_begin = jax.random.randint(rng_col, (1,), 0, image_shape[1] - mask_shape[1])

    # Find the coordinates for the opposite corner of the masked region
    row_end = row_begin + mask_shape[0]
    col_end = col_begin + mask_shape[1]

    # Generate a meshgrid with x and y coordinates of each pixel in the image
    img_cols, img_rows = jnp.meshgrid(
        jnp.arange(image_shape[1]), jnp.arange(image_shape[0])
    )

    # Create horizontal and vertical masks using boolean operations on the meshgrid coordinates
    img_rows = jnp.logical_or(img_rows < row_begin, img_rows >= row_end)
    img_cols = jnp.logical_or(img_cols < col_begin, img_cols >= col_end)

    # Combine the horizontal and vertical masks to create a rectangular mask
    img = jnp.float32(jnp.logical_or(img_rows, img_cols))

    # Find the fraction of the image that was masked
    # This should be close to the ratio parameter, but not exactly equal
    actual_ratio = 1 - mask_shape[0] * mask_shape[1] / (image_shape[0] * image_shape[1])

    return img[..., None], actual_ratio


def cutout(beta_params=[1.0, 1.0], mask_fn=rectangular_mask):
    """
    Return a composable function that applies cutout to a batch of images.

    Parameters:
    -----------
    beta_params: list or scalar
        The parameters for the beta distribution used to generate the cutout ratios.
        If a scalar is given, the same value is used for both parameters.

    mask_fn: function
        The function used to generate the cutout masks.

    Returns:
    --------
    Composable
        A composable function that applies cutout to a batch of images.
    """

    # If there is no beta_params, disable the augmentation
    if beta_params is None or jnp.all(beta_params <= 0):
        return identity

    # Convert beta_params to an array
    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]
    beta_params = jnp.array(beta_params, dtype=jnp.float32)

    @Composable
    def _cutout(values):
        """
        Apply cutout to a batch of images.

        Parameters:
        -----------
        values["inputs"]: jnp.ndarray
            The images to apply cutout to.
        values["rngs"]: jnp.ndarray
            The random keys.

        Returns:
        --------
        values["inputs"]: jnp.ndarray
            The images with cutout applied.
        values["rngs"]: jnp.ndarray
            The new random keys.
        """

        rngs = values["rngs"]
        xs = values["inputs"]

        # Split the random keys
        rngs, rngs_ratios, rngs_mask = batch_split(rngs, 3)

        # Sample the ratios from the beta distribution
        ratios = batch_beta(rngs_ratios, *beta_params)

        # Generate the cutout masks
        masks, actual_ratios = mask_fn(rngs_mask, xs.shape[1:], ratios)

        # Apply cutout
        xs = xs * masks

        return {**values, "inputs": xs, "rngs": rngs}

    return _cutout


def mixup(beta_params=[1.0, 1.0]):
    """
    Return a composable function that applies mixup to a batch of images.

    Parameters:
    -----------
    beta_params: list or scalar
        The parameters for the beta distribution used to generate the mixup ratios.
        If a scalar is given, the same value is used for both parameters.

    Returns:
    --------
    Composable
        A composable function that applies mixup to a batch of images.
    """

    # If there is no beta_params, disable the augmentation
    if beta_params is None or jnp.all(beta_params <= 0):
        return identity

    # Convert beta_params to an array
    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]
    beta_params = jnp.array(beta_params, dtype=jnp.float32)

    @Composable
    def _mixup(values):
        """
        Apply mixup to a batch of images.

        Parameters:
        -----------
        values["inputs"]: jnp.ndarray
            The images to apply mixup to.
        values["one_hot_labels"]: jnp.ndarray
            The labels to apply mixup to.
        values["rngs"]: jnp.ndarray
            The random keys.
        values["rng"]: jnp.ndarray
            The batch random key.

        Returns:
        --------
        values["inputs"]: jnp.ndarray
            The images with mixup applied.
        values["one_hot_labels"]: jnp.ndarray
            The labels with mixup applied.
        values["rngs"]: jnp.ndarray
            The new random keys.
        """

        rng = values["rng"]
        rngs = values["rngs"]
        xs = values["inputs"]
        ys = values["one_hot_labels"]

        # Split the random keys
        rngs, rngs_ratios = batch_split(rngs, 2)
        rng, rng_permutation = jax.random.split(rng, 2)

        # Sample the ratios from the beta distribution
        ratios = batch_beta(rngs_ratios, *beta_params)

        # Permute the images and labels
        idx = jax.random.permutation(rng_permutation, jnp.arange(len(xs)))
        xs2 = xs[idx]
        ys2 = ys[idx]

        # Apply mixup
        xs = xs * ratios[..., None, None, None] + xs2 * (
            1 - ratios[..., None, None, None]
        )
        ys = ys * ratios[..., None] + ys2 * (1 - ratios[..., None])

        return {**values, "inputs": xs, "one_hot_labels": ys, "rngs": rngs, "rng": rng}

    return _mixup


def cutmix(beta_params=[1.0, 1.0], mask_fn=rectangular_mask):
    """
    Return a composable function that applies cutmix to a batch of images.

    Parameters:
    -----------
    beta_params: list or scalar
        The parameters for the beta distribution used to generate the cutmix ratios.
        If a scalar is given, the same value is used for both parameters.

    mask_fn: function
        The function used to generate the cutmix masks.

    Returns:
    --------
    Composable
        A composable function that applies cutmix to a batch of images.
    """

    # If there is no beta_params, disable the augmentation
    if beta_params is None or jnp.all(beta_params <= 0):
        return identity

    # Convert beta_params to an array
    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]
    beta_params = jnp.array(beta_params, dtype=jnp.float32)

    @Composable
    def _cutmix(values):
        """
        Apply cutmix to a batch of images.

        Parameters:
        -----------
        values["inputs"]: jnp.ndarray
            The images to apply cutmix to.
        values["one_hot_labels"]: jnp.ndarray
            The labels to apply cutmix to.
        values["rngs"]: jnp.ndarray
            The random keys.
        values["rng"]: jnp.ndarray
            The batch random key.

        Returns:
        --------
        values["inputs"]: jnp.ndarray
            The images with cutmix applied.
        values["one_hot_labels"]: jnp.ndarray
            The labels with cutmix applied.
        values["rngs"]: jnp.ndarray
            The new random keys.
        values["rng"]: jnp.ndarray
            The new batch random key.
        """

        rng = values["rng"]
        rngs = values["rngs"]
        xs = values["inputs"]
        ys = values["one_hot_labels"]

        # Split the random keys
        rng, rng_permutation = jax.random.split(rng, 2)
        rngs, rngs_ratios, rngs_mask = batch_split(rngs, 3)

        # Sample the ratios from the beta distribution
        ratios = batch_beta(rngs_ratios, *beta_params)

        # Generate the cutout masks
        masks, actual_ratios = mask_fn(rngs_mask, xs.shape[1:3], ratios)

        # Permute the images and labels
        idx = jax.random.permutation(rng_permutation, jnp.arange(len(xs)))
        xs2 = xs[idx]
        ys2 = ys[idx]

        # Apply cutmix
        xs = xs * masks + xs2 * (1 - masks)
        ys = ys * actual_ratios[..., None] + ys2 * (1 - actual_ratios[..., None])

        return {**values, "inputs": xs, "one_hot_labels": ys, "rngs": rngs, "rng": rng}

    return _cutmix
