from functools import partial
from typing import Tuple, Union

import jax
from jax import numpy as jnp

from soundscape import utils

"""
Define vmapped versions of some functions in jax.random.

These are used because each sample in a batch has its own random key.
"""


_batch_uniform = jax.vmap(
    lambda rng, minval, maxval: jax.random.uniform(rng, minval=minval, maxval=maxval),
    in_axes=(0, None, None),
)

_batch_beta = jax.vmap(
    lambda rng, a, b: jax.random.beta(rng, a=a, b=b),
    in_axes=(0, None, None),
)


def _get_crop_arrays_fn(
    original_length: float, cropped_length: float, axis: int = -2
) -> callable:
    """
    Crop an array along the time axis (-2 by default).
    All length arguments are in seconds.
    """

    @jax.vmap
    def _crop_time_array(inputs: jax.Array, crop_times: jax.Array) -> jax.Array:
        duration_idx = int((cropped_length / original_length) * inputs.shape[axis])
        durations = inputs.shape[:axis] + (duration_idx,) + inputs.shape[axis + 1 :]

        begin_idx = (crop_times * inputs.shape[axis] / original_length).astype(int)

        begin_indices = [0] * len(inputs.shape)
        begin_indices[axis] = begin_idx

        return jax.lax.dynamic_slice(inputs, begin_indices, durations)

    return _crop_time_array


def _centered_crop_times(
    rngs: jax.Array, original_lenght: float, cropped_length: float
) -> jax.Array:
    """
    Generate crop times that crop the center of the array.
    """

    crop_time = (original_lenght - cropped_length) / 2
    return jnp.repeat(crop_time, len(rngs))


def _random_crop_times(
    rngs: jax.Array, original_length: float, cropped_length: float
) -> jax.Array:
    """
    Generate random crop times.
    """

    crop_times = _batch_uniform(rngs, 0, original_length - cropped_length)
    return crop_times


def crop_inputs(
    batch, crop_type: str, original_length: float, cropped_length: float, axis: int = -2
):
    """
    Return a function that applies crop augmentation to a batch of images.
    """

    crop_arrays_fn = _get_crop_arrays_fn(original_length, cropped_length, axis)
    crop_times_generator = (
        _random_crop_times if crop_type.upper() == "RANDOM" else _centered_crop_times
    )

    batch, _rngs = utils.split_rngs(batch)
    crop_times = crop_times_generator(_rngs, original_length, cropped_length)
    cropped_inputs = crop_arrays_fn(batch["inputs"], crop_times)

    return batch | {"inputs": cropped_inputs, "crop_times": crop_times}


@partial(jax.vmap, in_axes=(0, 0, None))
def _rectangular_mask(
    rng: jax.Array, approximate_ratio: float, image_shape
) -> (jax.Array, float):
    """
    Generate a mask matrix full of 1s, except for a rectangular region of 0s.
    The fraction of the image that is masked is roughly equal to the ratio parameter.

    Returns a (mask_matrix, actual_ratio) tuple.
    """

    rng, rng_row, rng_col = jax.random.split(rng, 3)

    mask_shape = jnp.int32(
        jnp.array(image_shape)[None, ...] * jnp.sqrt(1 - approximate_ratio)[..., None]
    )[0]

    # Generate coordinates for one corner of the masked region
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


def _preprocess_beta_params(
    beta_params: float | list[float] | None,
) -> list[float] | None:
    """
    Convert beta_params to a list if it is a float.
    """

    if beta_params is None or jnp.all(beta_params <= 0):
        return None

    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]

    if min(beta_params) < 0:
        return None

    return beta_params


def _permute(arrays: list[jax.Array], rng: jax.Array) -> list[jax.Array]:
    """
    Create a consistent random permutation of the given arrays.
    """

    idx = jax.random.permutation(rng, jnp.arange(len(arrays[0])))
    arrays = [array[idx] for array in arrays]
    return arrays


def _mix(
    array: jax.Array, array2: jax.Array | None, mixing: jax.Array | float
) -> jax.Array:
    """
    Mix two arrays together using the given mixing mask.
    If array2 is None, 0 values are used.
    """
    if isinstance(mixing, float):
        mixing = jnp.repeat(mixing, len(array))

    if len(mixing.shape) == 1:
        array_rank = len(array.shape)
        mixing = mixing.reshape((len(mixing),) + (1,) * (array_rank - 1))

    if array2 is None:
        return array * mixing
    return array * mixing + array2 * (1 - mixing)


def cutout(batch, beta_params, mask_fn=_rectangular_mask):
    """
    Return a function that applies cutout to a batch of images.
    """

    beta_params = _preprocess_beta_params(beta_params)

    if beta_params is None:
        return lambda batch: batch

    batch, rngs_ratios, rngs_mask = utils.split_rngs(batch, 2)

    ratios = _batch_beta(rngs_ratios, *beta_params)
    masks, _ = mask_fn(rngs_mask, ratios, batch["inputs"].shape[1:])

    inputs = _mix(batch["inputs"], None, masks)

    return batch | {"inputs": inputs}


def mixup(batch, beta_params):
    """
    Apply mixup to a batch of images.
    """

    beta_params = _preprocess_beta_params(beta_params)
    if beta_params is None:
        return batch

    batch, rngs = utils.split_rngs(batch)
    batch, rng = utils.split_rng(batch)

    ratios = _batch_beta(rngs, *beta_params)

    inputs2, label_probs2 = _permute([batch["inputs"], batch["label_probs"]], rng)
    inputs = _mix(batch["inputs"], inputs2, ratios)
    label_probs = _mix(batch["label_probs"], label_probs2, ratios)

    return batch | {"inputs": inputs, "label_probs": label_probs}


def cutmix(batch, beta_params, mask_fn=_rectangular_mask):
    """
    Return a function that applies cutmix to a batch of images.
    """

    beta_params = _preprocess_beta_params(beta_params)

    if beta_params is None:
        return lambda batch: batch

    batch, rngs_ratios, rngs_mask = utils.split_rngs(batch, 2)
    batch, rng = utils.split_rng(batch)

    ratios = _batch_beta(rngs_ratios, *beta_params)
    masks, actual_ratios = mask_fn(rngs_mask, ratios, batch["inputs"].shape[1:3])

    inputs2, label_probs2 = _permute([batch["inputs"], batch["label_probs"]], rng)
    inputs = _mix(batch["inputs"], inputs2, masks)
    label_probs = _mix(batch["label_probs"], label_probs2, actual_ratios)

    return batch | {"inputs": inputs, "label_probs": label_probs}
