from jax import numpy as jnp
import jax
from typing import Protocol, Union
from jax import random

from . import settings
from .composition import Composable

settings_fn, settings_dict = settings.from_file("dataset_settings.yaml")


class Augmentation(Protocol):
    def __call__(self, rngs: jax.random.PRNGKey, xs: jnp.ndarray) -> jnp.ndarray:
        ...


def augment_batch(augment_fns: Union[list[Augmentation], Augmentation]) -> Composable:
    if not isinstance(augment_fns, list):
        augment_fns = [augment_fns]

    @Composable
    def _augment(values):
        xs = values["inputs"]
        rngs = values["rng"]

        rngs = random.split(rngs)
        rngs, augment_rngs = rngs[:, 0], rngs[:, 1:]

        for augment_fn in augment_fns:
            xs = augment_fn(augment_rngs, xs)

        return {**values, "inputs": xs, "rng": rngs}

    return _augment


def crop_time_array(array, duration, begin_time, end_time, axis=1):
    begin_idx = int(begin_time * array.shape[axis] / duration)
    end_idx = int(end_time * array.shape[axis] / duration)
    return jnp.take(array, jnp.arange(begin_idx, end_idx), axis=axis)


@settings_fn
def deterministic_time_crop(*, segment_length, cropped_length, extension):
    axis = 2 if extension == "png" else 1

    def _deterministic_time_crop(rng, xs):
        begin_time = (segment_length - cropped_length) / 2
        end_time = begin_time + cropped_length
        return crop_time_array(xs, segment_length, begin_time, end_time, axis)

    return _deterministic_time_crop


@settings_fn
def random_time_crop(*, segment_length, cropped_length):
    def _random_time_crop(rng, xs):
        begin_time = random.uniform(rng, (0, segment_length - cropped_length))
        end_time = begin_time + cropped_length
        return crop_time_array(xs, segment_length, begin_time, end_time)

    return _random_time_crop


def all_possible_augmentations(augment_fns: list[Augmentation]) -> Augmentation:
    def _augment(rng, x):
        rngs = jax.random.split(rng, len(augment_fns))
        return jnp.array([augment(_rng, x) for augment, _rng in zip(augment_fns, rngs)])

    return _augment


def augmix(possible_augmentation_fns, depth, width, dirichlet_alpha):
    """ """

    def augment_once(rng, x):
        d = jax.random.randint(rng, (1,), 0, depth)
        rngs = jax.random.split(rng, d)

        for rng in rngs:
            rng_fn, rng_aug = jax.random.split(rng)
            fn = jax.random.choice(rng_fn, possible_augmentation_fns)
            x = fn(rng_aug, x)

        return x

    def mix(rng, xs):
        mixing_weights = jax.random.dirichlet(rng, (dirichlet_alpha,) * len(xs))
        return (xs * mixing_weights).sum(axis=0)

    def _augment(rng, x):
        rngs = jax.random.split(rng, width + 1)
        return mix(rngs[0], jnp.array([augment_once(rng, x) for rng in rngs[1:]]))

    return _augment
