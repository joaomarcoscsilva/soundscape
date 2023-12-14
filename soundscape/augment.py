from functools import partial
from typing import Protocol, Union

import jax
import numpy as jnp

from .composition import Composable, StateFunction, identity
from .settings import settings_fn
from .typechecking import Array

"""
Define vmapped versions of some functions in jax.random.

These are used because each sample in a batch has its own random key.
"""

batch_rng_split = jax.vmap(
    lambda rng, n: tuple(jax.random.split(rng, n)), in_axes=(0, None)
)

batch_uniform = jax.vmap(
    lambda rng, minval, maxval: jax.random.uniform(rng, minval=minval, maxval=maxval),
    in_axes=(0, None, None),
)

batch_beta = jax.vmap(
    lambda rng, a, b: jax.random.beta(rng, a=a, b=b),
    in_axes=(0, None, None),
)


def _crop_arrays(
    original_length: float, cropped_length: float, axis: int
) -> StateFunction:
    """
    Crop an array along the time axis.
    """

    axis = axis - 1

    @jax.vmap
    @StateFunction.with_output("inputs", typecheck=False)
    def _crop_time_array(inputs: Array, crop_times: Array) -> Array:
        duration_idx = int((cropped_length / original_length) * inputs.shape[axis])
        durations = inputs.shape[:axis] + (duration_idx,) + inputs.shape[axis + 1 :]

        begin_idx = (crop_times * inputs.shape[axis] / original_length).astype(int)

        begin_indices = [0] * len(inputs.shape)
        begin_indices[axis] = begin_idx

        return jax.lax.dynamic_slice(inputs, begin_indices, durations)

    return _crop_time_array


def _centered_crop_times(
    original_lenght: float, cropped_length: float
) -> StateFunction:
    """
    Generate crop times that crop the center of the array.
    """

    @StateFunction.with_output("crop_times")
    def _crop_times(inputs: Array) -> Array:
        crop_time = (original_lenght - cropped_length) / 2
        return jnp.repeat(crop_time, len(inputs))

    return _crop_times


def _random_crop_times(original_length: float, cropped_length: float) -> StateFunction:
    """
    Generate random crop times.
    """

    @StateFunction
    def _crop_times(inputs: Array, rngs: Array) -> dict[str, Array]:
        rngs, _rngs = batch_rng_split(rngs, 2)
        crop_times = batch_uniform(_rngs, 0, original_length - cropped_length)
        return {"crop_times": crop_times, "rngs": rngs}

    return _crop_times


def crop_inputs(
    original_length: float, cropped_length: float, input_type: str, crop_type: str
) -> Composable:
    """
    Return a function that applies crop augmentation to a batch of images.
    """

    if crop_type == "center":
        generate_crop_times = _centered_crop_times
    elif crop_type == "random":
        generate_crop_times = _random_crop_times
    else:
        raise ValueError(f"Unknown crop type: {crop_type}")
    generate_crop_times = generate_crop_times(original_length, cropped_length)

    if input_type == "image":
        crop_axis = 1
    elif input_type == "audio":
        crop_axis = 0
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    crop_arrays = _crop_arrays(original_length, cropped_length, crop_axis)

    return generate_crop_times | crop_arrays


@partial(jax.vmap, in_axes=(0, None, 0))
def rectangular_mask(rng: Array, image_shape: tuple[int], ratio: float) -> Array:
    """
    Generate a mask matrix full of 1s, except for a rectangular region of 0s.
    The fraction of the image that is masked is roughly equal to the ratio parameter.

    Returns a (mask_matrix, actual_ratio) tuple.
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


def preprocess_beta_params(
    beta_params: float | list[float] | None,
) -> list[float] | None:
    """
    Convert beta_params to a list if it is a float.
    """

    if beta_params is None or jnp.all(beta_params <= 0):
        return None

    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]

    return beta_params


def permute(arrays: list[Array], rng: Array) -> list[Array]:
    """
    Create a consistent random permutation of the given arrays.
    """

    idx = jax.random.permutation(rng, jnp.arange(len(arrays[0])))
    arrays = [array[idx] for array in arrays]
    return arrays


def mix(array: Array, array2: Array | None, mixing: Array | float) -> Array:
    """
    Mix two arrays together using the given mixing mask.
    If array2 is None, 0 values are used.
    """

    if array2 is None:
        return array * mixing
    return array * mixing + array2 * (1 - mixing)


def cutout(beta_params=[1.0, 1.0], mask_fn=rectangular_mask) -> StateFunction:
    """
    Return a function that applies cutout to a batch of images.
    """

    beta_params = preprocess_beta_params(beta_params)

    @StateFunction
    def _cutout(inputs, rngs):
        """
        Apply cutout to a batch of images.
        """

        rngs, rngs_ratios, rngs_mask = batch_rng_split(rngs, 3)
        ratios = batch_beta(rngs_ratios, *beta_params)
        masks, actual_ratios = mask_fn(rngs_mask, xs.shape[1:], ratios)

        inputs = mix(inputs, None, masks)

        return {"inputs": inputs, "rngs": rngs}

    return _cutout


def mixup(beta_params: float | list[float] | None = 1.0) -> StateFunction:
    """
    Return a function that applies mixup to a batch of images.
    """

    beta_params = preprocess_beta_params(beta_params)

    @StateFunction
    def _mixup(
        inputs: Array, label_probs: Array, rngs: Array, rng: Array
    ) -> dict[str, Array]:
        rngs, rngs_ratios = batch_rng_split(rngs, 2)
        ratios = batch_beta(rngs_ratios, *beta_params)

        rng, rng_permutation = jax.random.split(rng, 2)
        inputs2, label_probs2 = permute([inputs, label_probs], rng_permutation)

        inputs = mix(inputs, inputs2, ratios[..., None, None, None])
        label_probs = mix(label_probs, label_probs2, ratios[..., None])

        return {"inputs": inputs, "label_probs": label_probs, "rngs": rngs, "rng": rng}

    return _mixup


def cutmix(beta_params=[1.0, 1.0], mask_fn=rectangular_mask) -> Composable:
    """
    Return a function that applies cutmix to a batch of images.
    """

    # If there is no beta_params, disable the augmentation
    if beta_params is None or jnp.all(beta_params <= 0):
        return identity

    # Convert beta_params to an array
    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]
    beta_params = jnp.array(beta_params, dtype=jnp.float32)

    @StateFunction
    def _cutmix(
        inputs: Array, label_probs: Array, rngs: Array, rng: Array
    ) -> dict[str, Array]:
        rngs, rngs_ratios, rngs_mask = batch_rng_split(rngs, 3)
        ratios = batch_beta(rngs_ratios, *beta_params)
        masks, actual_ratios = mask_fn(rngs_mask, xs.shape[1:3], ratios)

        rng, rng_permutation = jax.random.split(rng, 2)
        inputs2, label_probs2 = permute([inputs, label_probs], rng_permutation)

        inputs = mix(inputs, inputs2, masks)
        label_probs = mix(label_probs, label_probs2, actual_ratios[..., None])

        return {**values, "inputs": xs, "one_hot_labels": ys, "rngs": rngs, "rng": rng}

    return _cutmix
