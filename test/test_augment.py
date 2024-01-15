from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp

from soundscape import augment


def assert_all_different(values_list):
    for i in range(len(values_list)):
        for j in range(i + 1, len(values_list)):
            assert not jnp.allclose(values_list[i], values_list[j])


def assert_all_close(values_list):
    for i in range(len(values_list)):
        for j in range(i + 1, len(values_list)):
            assert jnp.allclose(values_list[i], values_list[j])


def assert_equal_channels(image_batch):
    assert jnp.allclose(image_batch[..., 0], image_batch[..., 1])
    assert jnp.allclose(image_batch[..., 0], image_batch[..., 2])


def assert_all_constant(image_batch):
    flat_image_batch = image_batch.reshape(len(image_batch), -1)
    assert jnp.allclose(flat_image_batch.min(1), flat_image_batch.max(1))


def assert_nondeterministic_probabilities(label_probs_batch):
    assert 0 <= label_probs_batch.max() <= 1
    assert jnp.allclose(label_probs_batch.sum(1), 1)

    assert label_probs_batch.std(1).max() > 0
    assert label_probs_batch.std(0).max() > 0


def assert_all_rectangular(image_batch, bg_colors):
    """
    Assert that all images in the batch are a solid background color
    with a solid rectangular shape inside.
    """

    if isinstance(bg_colors, int):
        bg_colors = jnp.repeat(bg_colors, len(image_batch))

    for i, image in enumerate(image_batch):
        mask = image != bg_colors[i]

        if mask.any():
            nonzero_xs, nonzero_ys = jnp.nonzero(mask[..., 0])
            min_row, max_row = nonzero_xs.min(), nonzero_xs.max()
            min_col, max_col = nonzero_ys.min(), nonzero_ys.max()

            assert jnp.allclose(mask[min_row : max_row + 1, min_col : max_col + 1], 1)


def test_centered_crops():
    batch = {"inputs": jnp.arange(500)[None, ...]}
    key1 = {"rng": jax.random.PRNGKey(0)}
    key2 = {"rng": jax.random.PRNGKey(1)}

    cropped = augment.crop_inputs(batch | key1, "center", 5, 3, 0)["inputs"]
    cropped2 = augment.crop_inputs(batch | key2, "center", 5, 3, 0)["inputs"]

    assert_all_close([cropped, cropped2, jnp.arange(100, 400)[None, ...]])


def test_random_crops():
    batch = {"inputs": jnp.arange(500)[None, ...]}
    key1 = {"rng": jax.random.PRNGKey(0)}
    key2 = {"rng": jax.random.PRNGKey(1)}

    cropped = augment.crop_inputs(batch | key1, "random", 5, 3, 0)["inputs"]
    cropped2 = augment.crop_inputs(batch | key2, "random", 5, 3, 0)["inputs"]

    assert cropped.shape == cropped2.shape == (1, 300)
    assert not jnp.allclose(cropped, cropped2)


def test_cutout():
    # The original image batch contains images with all pixels set to 2
    batch = {"inputs": jnp.ones((64, 100, 100, 3)) * 2}
    key1 = {"rng": jax.random.PRNGKey(0)}
    key2 = {"rng": jax.random.PRNGKey(1)}

    cut = augment.cutout(batch | key1, 1)["inputs"]
    cut2 = augment.cutout(batch | key2, 1)["inputs"]

    assert not jnp.allclose(cut, cut2)

    assert cut.shape == batch["inputs"].shape
    assert cut.min() == 0
    assert cut.max() == 2

    assert jnp.allclose(cut.mean(), 1.0, atol=0.2)

    assert_equal_channels(cut)
    assert_all_rectangular(cut[:8], 2)
    assert_all_different(cut[:32])


def test_mixup():
    num_classes = 5
    batch = {
        "inputs": jnp.ones((64, 100, 100, 3)) * jnp.arange(64).reshape(-1, 1, 1, 1),
        "label_probs": jax.nn.one_hot(jnp.arange(64) % num_classes, num_classes),
    }
    key1 = {"rng": jax.random.PRNGKey(0)}
    key2 = {"rng": jax.random.PRNGKey(1)}

    mixed = augment.mixup(batch | key1, 1.0)
    mixed2 = augment.mixup(batch | key2, 1.0)

    assert not jnp.allclose(mixed["inputs"], mixed2["inputs"])

    assert mixed["inputs"].shape == batch["inputs"].shape
    assert mixed["inputs"].min() >= 0
    assert mixed["inputs"].max() <= 63

    assert jnp.allclose(mixed["inputs"].mean(), batch["inputs"].mean(), atol=0.2)

    assert_all_different(mixed["inputs"])
    assert_all_constant(mixed["inputs"])

    assert not jnp.allclose(mixed["label_probs"], mixed2["label_probs"])

    assert mixed["label_probs"].shape == batch["label_probs"].shape
    assert mixed["label_probs"].min() == 0

    assert_nondeterministic_probabilities(mixed["label_probs"])


def test_cutmix():
    num_classes = 5
    batch = {
        "inputs": jnp.ones((512, 100, 100, 3)) * jnp.arange(512).reshape(-1, 1, 1, 1),
        "label_probs": jax.nn.one_hot(jnp.arange(512) % num_classes, num_classes),
    }
    key1 = {"rng": jax.random.PRNGKey(0)}
    key2 = {"rng": jax.random.PRNGKey(1)}

    mixed = augment.cutmix(batch | key1, 1.0)
    mixed2 = augment.cutmix(batch | key2, 1.0)

    assert not jnp.allclose(mixed["inputs"], mixed2["inputs"])

    assert mixed["inputs"].shape == batch["inputs"].shape
    assert mixed["inputs"].min() >= 0
    assert mixed["inputs"].max() <= 511

    assert jnp.allclose(mixed["inputs"].mean(), batch["inputs"].mean(), atol=1)

    assert_equal_channels(mixed["inputs"])
    assert_all_rectangular(mixed["inputs"][:8], jnp.arange(512))
    assert_all_different(mixed["inputs"][:64])

    assert not jnp.allclose(mixed["label_probs"], mixed2["label_probs"])

    assert mixed["label_probs"].shape == batch["label_probs"].shape
    assert mixed["label_probs"].min() == 0

    assert_nondeterministic_probabilities(mixed["label_probs"])
