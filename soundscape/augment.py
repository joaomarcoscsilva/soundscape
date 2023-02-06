from jax import numpy as jnp
import jax
from typing import Protocol, Union
from jax import random
from functools import partial

from .settings import settings_fn
from .composition import Composable, identity


batch_split = jax.vmap(lambda rng, n: tuple(random.split(rng, n)), in_axes=(0, None))

batch_uniform = jax.vmap(
    lambda rng, minval, maxval: random.uniform(rng, minval=minval, maxval=maxval),
    in_axes=(0, None, None),
)

batch_beta = jax.vmap(
    lambda rng, a, b: random.beta(rng, a=a, b=b),
    in_axes=(0, None, None),
)


@partial(jax.vmap, in_axes=(0, None, None, 0, None))
def crop_time_array(array, segment_length, cropped_length, begin_time, axis=1):

    axis = axis - 1  # change the axis because of the batch dimension

    begin_idx = jnp.int32(begin_time * array.shape[axis] / segment_length)
    duration_idx = jnp.int32((cropped_length / segment_length) * array.shape[axis])

    begin_indices = [0] * len(array.shape)
    durations = list(array.shape)

    begin_indices[axis] = begin_idx
    durations[axis] = duration_idx

    return jax.lax.dynamic_slice(array, begin_indices, durations)


@settings_fn
def deterministic_time_crop(values, *, segment_length, cropped_length, extension):
    axis = 2 if extension == "png" else 1

    begin_time = (segment_length - cropped_length) / 2
    begin_times = jnp.repeat(begin_time, len(values["inputs"]))

    new_inputs = crop_time_array(
        values["inputs"], segment_length, cropped_length, begin_times, axis
    )

    return {**values, "inputs": new_inputs}


@settings_fn
def random_time_crop(values, *, segment_length, cropped_length, extension):
    axis = 2 if extension == "png" else 1

    rngs, _rngs = batch_split(values["rngs"], 2)

    begin_times = batch_uniform(_rngs, 0, segment_length - cropped_length)

    new_inputs = crop_time_array(
        values["inputs"], segment_length, cropped_length, begin_times, axis
    )

    return {**values, "inputs": new_inputs, "rngs": rngs}


@settings_fn
def time_crop(crop_type="deterministic"):
    if crop_type == "deterministic":
        return deterministic_time_crop
    elif crop_type == "random":
        return random_time_crop
    else:
        return identity


def augmix(possible_augmentation_fns, depth, width, dirichlet_alpha):
    def augment(rng, x):
        _depth = jax.random.randint(rng, (1,), 0, depth)
        rngs = jax.random.split(rng, _depth)

        for rng in rngs:
            rng_fn, rng_aug = jax.random.split(rng)
            fn = jax.random.choice(rng_fn, possible_augmentation_fns)
            x = fn(rng_aug, x)

        return x

    def mix(rng, xs):
        mixing_weights = jax.random.dirichlet(rng, (dirichlet_alpha,) * len(xs))
        return (xs * mixing_weights).sum(axis=0)

    def augment_and_mix(rng, x):
        rng_mix, rng_augment = jax.random.split(rng, width + 1)
        xs = jnp.array([augment(rng_augment, x) for _ in range(width)])
        return mix(rng_mix, xs)

    def _augmix(values):
        rngs, _rngs1, _rngs2 = batch_split(values["rngs"], 3)

        xs = values["inputs"]

        new_xs_1 = [augment(rng, x) for rng, x in zip(_rngs1, xs)]
        new_xs_1 = jnp.stack(new_xs_1)

        new_xs_2 = [augment_and_mix(rng, x) for rng, x in zip(_rngs2, xs)]
        new_xs_2 = jnp.stack(new_xs_2)

        return {
            **values,
            "inputs": jnp.concatenate([xs, new_xs_1, new_xs_2], axis=0),
            "rngs": rngs,
        }

    return _augmix


@partial(jax.vmap, in_axes=(0, None, 0))
def rectangular_mask(rng, image_shape, ratio):
    rng, rng_row, rng_col = jax.random.split(rng, 3)

    mask_shape = jnp.int32(jnp.array(image_shape) * jnp.sqrt(1 - ratio))

    row_begin = jax.random.randint(rng_row, (1,), 0, image_shape[0] - mask_shape[0])
    col_begin = jax.random.randint(rng_col, (1,), 0, image_shape[1] - mask_shape[1])

    row_end = row_begin + mask_shape[0]
    col_end = col_begin + mask_shape[1]

    img_cols, img_rows = jnp.meshgrid(
        jnp.arange(image_shape[1]), jnp.arange(image_shape[0])
    )

    img_rows = jnp.logical_or(img_rows < row_begin, img_rows >= row_end)
    img_cols = jnp.logical_or(img_cols < col_begin, img_cols >= col_end)

    img = jnp.float32(jnp.logical_or(img_rows, img_cols))

    return img[..., None]


def cutout(beta_params=[1.0, 1.0], mask_fn=rectangular_mask):
    if beta_params is None:
        return identity
    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]

    beta_params = jnp.array(beta_params, dtype=jnp.float32)

    @Composable
    def _cutout(values):
        rngs = values["rngs"]
        xs = values["inputs"]

        rngs, rngs_ratios, rngs_mask = batch_split(rngs, 3)

        ratios = batch_beta(rngs_ratios, *beta_params)

        masks = mask_fn(rngs_mask, xs.shape[1:3], ratios)

        xs = xs * masks

        return {**values, "inputs": xs, "rngs": rngs}

    return _cutout


def mixup(beta_params=[1.0, 1.0]):
    if beta_params is None:
        return identity
    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]

    beta_params = jnp.array(beta_params, dtype=jnp.float32)

    @Composable
    def _mixup(values):
        rngs = values["rngs"]
        xs = values["inputs"]
        ys = values["one_hot_labels"]

        rngs, rngs_ratios, rngs_permutation = batch_split(rngs, 3)

        ratios = batch_beta(rngs_ratios, *beta_params)

        idx = jax.random.permutation(rngs_permutation[0], jnp.arange(len(xs)))
        xs2 = xs[idx]
        ys2 = ys[idx]

        xs = xs * ratios[..., None, None, None] + xs2 * (
            1 - ratios[..., None, None, None]
        )
        ys = ys * ratios[..., None] + ys2 * (1 - ratios[..., None])

        return {**values, "inputs": xs, "one_hot_labels": ys, "rngs": rngs}

    return _mixup


def cutmix(beta_params=[1.0, 1.0], mask_fn=rectangular_mask):
    if beta_params is None:
        return identity
    if type(beta_params) is not list:
        beta_params = [beta_params, beta_params]

    beta_params = jnp.array(beta_params, dtype=jnp.float32)

    @Composable
    def _cutmix(values):
        rngs = values["rngs"]
        xs = values["inputs"]
        ys = values["one_hot_labels"]

        rngs, rngs_ratios, rngs_mask, rngs_permutation = batch_split(rngs, 4)

        ratios = batch_beta(rngs_ratios, *beta_params)

        masks = mask_fn(rngs_mask, xs.shape[1:3], ratios)

        idx = jax.random.permutation(rngs_permutation[0], jnp.arange(len(xs)))
        xs2 = xs[idx]
        ys2 = ys[idx]

        xs = xs * masks + xs2 * (1 - masks)
        ys = ys * ratios[..., None] + ys2 * (1 - ratios[..., None])

        return {**values, "inputs": xs, "one_hot_labels": ys, "rngs": rngs}

    return _cutmix
