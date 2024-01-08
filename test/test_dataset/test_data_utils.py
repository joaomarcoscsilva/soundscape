import jax
from jax import numpy as jnp

from soundscape.dataset import data_utils


def test_prepare_image():
    rng = jax.random.PRNGKey(0)
    img = jax.random.randint(rng, (10, 1000, 128, 1), 0, 2**16 - 1, dtype=jnp.uint16)

    batch = {"inputs": img}
    batch2 = data_utils.prepare_images(batch)
    img2 = batch2["inputs"]

    assert img2.shape == (10, 1000, 128, 3)
    assert img2.dtype == jnp.float32
    assert img2.min() >= 0
    assert img2.max() <= 1

    assert jnp.allclose(img2.mean(), 0.5, atol=0.01)
    assert jnp.allclose(img2[..., 0], img2[..., 1], atol=0.01)
    assert jnp.allclose(img2[..., 0], img2[..., 2], atol=0.01)
    assert jnp.allclose(img2[..., 0], img[..., 0] / 2**16, atol=0.01)


def test_one_hot_encode():
    labels = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]
    )

    batch = {"labels": labels}
    batch2 = data_utils.one_hot_encode(batch, num_classes=10)

    assert jnp.allclose(batch2["label_probs"][0], jnp.eye(10))
    assert jnp.allclose(batch2["label_probs"][1], jnp.eye(10)[::-1])
    assert jnp.allclose(batch2["labels"], batch["labels"])


def test_downsample_image():
    rng = jax.random.PRNGKey(0)
    img = jax.random.randint(rng, (10, 1000, 128, 2), 0, 2**16 - 1, dtype=jnp.uint16)

    batch = {"inputs": img}
    batch2 = data_utils.downsample_images(batch)
    img2 = batch2["inputs"]

    assert img2.shape == (10, 224, 224, 2)
    assert jnp.linalg.matrix_rank(img2.reshape(10, -1)) == 10
