from jax import random, numpy as jnp
from . import dataset, augment

rng = random.PRNGKey(0)
ds = dataset.get_dataset("train", rng)

augment_fn = augment.augment_batch([augment.deterministic_time_crop()])

pipeline = (
    dataset.tf2jax
    | dataset.prepare_image
    | dataset.downsample_image
    | dataset.prepare_image_channels
    | augment_fn
)

for i in ds:
    i = pipeline(i)
    break
