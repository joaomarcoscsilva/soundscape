from jax import numpy as jnp
import jax
from typing import Protocol, Union
from jax import random
from functools import partial

from . import settings
from .composition import Composable, identity

settings_fn, settings_dict = settings.from_file()


batch_split = jax.vmap(lambda rng, n: tuple(random.split(rng, n)), in_axes=(0, None))


def crop_time_array(array, duration, begin_time, end_time, axis=1):
    begin_idx = int(begin_time * array.shape[axis] / duration)
    end_idx = int(end_time * array.shape[axis] / duration)
    return jnp.take(array, jnp.arange(begin_idx, end_idx), axis=axis)


@settings_fn
def time_crop(*, segment_length, cropped_length, extension, crop_type="deterministic"):
    axis = 2 if extension == "png" else 1

    def _deterministic_time_crop(values):
        begin_time = (segment_length - cropped_length) / 2
        end_time = begin_time + cropped_length

        new_inputs = crop_time_array(
            values["inputs"], segment_length, begin_time, end_time, axis=axis
        )

        return {**values, "inputs": new_inputs}

    def _random_time_crop(values):
        rngs, _rngs = batch_split(values["rngs"])
        begin_time = random.uniform(_rngs, (0, segment_length - cropped_length))
        end_time = begin_time + cropped_length

        new_inputs = crop_time_array(
            values["inputs"], segment_length, begin_time, end_time, axis=axis
        )

        return {**values, "inputs": new_inputs, "rngs": rngs}

    if crop_type == "deterministic":
        return _deterministic_time_crop
    elif crop_type == "random":
        return _random_time_crop
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

    mask_shape = jnp.array([image_shape[0] * ratio, image_shape[1] * ratio])

    row_begin = jax.random.randint(rng_row, (1,), 0, image_shape[0] - mask_shape[0])
    col_begin = jax.random.randint(rng_col, (1,), 0, image_shape[1] - mask_shape[1])

    row_end = row_begin + mask_shape[0]
    col_end = col_begin + mask_shape[1]

    img_rows, img_cols = jnp.meshgrid(
        jnp.arange(image_shape[1]), jnp.arange(image_shape[0])
    )

    img_rows = jnp.logical_and(img_rows >= row_begin, img_rows < row_end)
    img_cols = jnp.logical_and(img_cols >= col_begin, img_cols < col_end)

    img = jnp.float32(jnp.logical_and(img_rows, img_cols))

    return img


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

        ratios = jax.vmap(jax.random.beta, in_axes=(0, None, None))(
            rngs_ratios, *beta_params
        )

        masks = mask_fn(rngs_mask, xs.shape[1:3], ratios)

        xs = xs * masks[..., None]

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

        ratios = jax.vmap(jax.random.beta, in_axes=(0, None, None))(
            rngs_ratios, *beta_params
        )

        idx = jax.random.permutation(rngs_permutation[0], jnp.arange(len(xs)))
        xs2 = xs[idx]
        ys2 = ys[idx]

        xs = xs2 * ratios[..., None, None, None] + xs * (
            1 - ratios[..., None, None, None]
        )
        ys = ys2 * ratios[..., None] + ys * (1 - ratios[..., None])

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

        ratios = jax.vmap(jax.random.beta, in_axes=(0, None, None))(
            rngs_ratios, *beta_params
        )
        masks = mask_fn(rngs_mask, xs.shape[1:3], ratios)

        idx = jax.random.permutation(rngs_permutation[0], jnp.arange(len(xs)))
        xs2 = xs[idx]
        ys2 = ys[idx]

        # breakpoint()

        xs = xs2 * masks[..., None] + xs * (1 - masks[..., None])
        ys = ys2 * ratios[..., None] + ys * (1 - ratios[..., None])

        return {**values, "inputs": xs, "one_hot_labels": ys, "rngs": rngs}

    return _cutmix
