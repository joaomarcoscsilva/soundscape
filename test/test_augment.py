from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp

from soundscape import augment, composition, settings
from soundscape.composition import State


def assert_all_different(values_list):
    for i in range(len(values_list)):
        for j in range(i + 1, len(values_list)):
            assert not jnp.allclose(values_list[i], values_list[j])


def test_batch_rngs():
    rng1 = jax.random.PRNGKey(0)
    rng2 = jax.random.PRNGKey(1)

    rngs1 = jax.random.split(rng1, 10)
    rngs2 = jax.random.split(rng2, 10)

    split_rngs1, _rngs1 = augment._batch_rng_split(rngs1, 2)
    split_rngs2, _rngs2 = augment._batch_rng_split(rngs2, 2)

    assert_all_different([split_rngs1, split_rngs2, _rngs1, _rngs2, rngs1, rngs2])

    uniform_1 = augment._batch_uniform(_rngs1, 0, 10)
    uniform_2 = augment._batch_uniform(_rngs2, 0, 10)

    assert_all_different([uniform_1, uniform_2])

    assert jnp.all(uniform_1 < 10)
    assert jnp.all(uniform_2 < 10)

    assert jnp.all(uniform_1 >= 0)
    assert jnp.all(uniform_2 >= 0)


def test_crop_arrays():
    batch = State({"inputs": jnp.arange(60)[None, ...], "crop_times": jnp.array([2])})

    cropped_batch = augment._crop_arrays(
        original_length=60.0, cropped_length=6.0, axis=1
    )(batch)

    assert jnp.allclose(cropped_batch["inputs"], jnp.arange(2, 8))

    batch["inputs"] = jnp.arange(60).reshape(1, 1, 60, 1)
    cropped_batch = augment._crop_arrays(
        original_length=60.0, cropped_length=6.0, axis=2
    )(batch)

    assert jnp.allclose(cropped_batch["inputs"], jnp.arange(2, 8).reshape(1, 1, 6, 1))


def test_crop_times():
    original_length = 20.0
    cropped_length = 12.0

    crop_times = augment.centered_crop_times(
        original_lenght=original_length, cropped_length=cropped_length
    )(jnp.arange(10))

    assert jnp.allclose(crop_times, jnp.repeat(4.0, 10))

    crop_times, new_rngs = augment.random_crop_times(
        original_length=original_length, cropped_length=cropped_length
    )(jnp.arange(10), jax.random.split(jax.random.key(0), 10))

    assert jnp.all(crop_times < 8.0)
    assert jnp.all(crop_times >= 0.0)
    assert crop_times.shape == (10,)
    assert crop_times.std() > 0.1
    assert new_rngs.shape == (10,)
    assert not jnp.allclose(new_rngs, jax.random.split(jax.random.key(0), 10))


def test_deterministic_time_crop():
    arr = jnp.arange(120).reshape(1, 2, 60, 1)
    crop_fn = augment.deterministic_time_crop(
        segment_length=60, cropped_length=4, extension="png"
    )
    new_arr = crop_fn({"inputs": arr})["inputs"]

    assert jnp.allclose(
        new_arr,
        jnp.array([[28, 29, 30, 31], [88, 89, 90, 91]]).reshape((1, 2, -1, 1)),
    )

    arr = jnp.arange(100).reshape((2, 50))
    crop_fn = augment.deterministic_time_crop(
        segment_length=5, cropped_length=3, extension="wav"
    )
    new_arr = crop_fn({"inputs": arr})["inputs"]

    assert jnp.allclose(
        new_arr,
        jnp.array([range(10, 40), range(60, 90)]),
    )


def test_random_time_crop():
    with settings.Settings(
        {"segment_length": 60, "cropped_length": 4, "extension": "png"}
    ):
        rng1 = jax.random.PRNGKey(0)[None, ...]
        rng2 = jax.random.PRNGKey(1)[None, ...]

        arr = jnp.arange(120).reshape(1, 2, 60, 1)
        new_values_1 = augment.random_time_crop()({"inputs": arr, "rngs": rng1})
        new_values_1_copy = augment.random_time_crop()({"inputs": arr, "rngs": rng1})
        new_values_2 = augment.random_time_crop()({"inputs": arr, "rngs": rng2})

        assert_all_different([rng1, rng2, new_values_1["rngs"], new_values_2["rngs"]])
        assert jnp.allclose(new_values_1["rngs"], new_values_1_copy["rngs"])

        new_arr_1 = new_values_1["inputs"]
        new_arr_1_copy = new_values_1_copy["inputs"]
        new_arr_2 = new_values_2["inputs"]

        assert jnp.allclose(new_arr_1, new_arr_1_copy)
        assert not jnp.allclose(new_arr_1, new_arr_2)
        assert new_arr_1.shape == (1, 2, 4, 1)

    with settings.Settings(
        {"segment_length": 5, "cropped_length": 3, "extension": "wav"}
    ):
        rng1 = jax.random.PRNGKey(0)
        rng2 = jax.random.PRNGKey(1)
        rng1 = jax.random.split(rng1, 2)
        rng2 = jax.random.split(rng2, 2)

        arr = jnp.arange(100).reshape((2, 50))
        new_arr_1 = augment.random_time_crop()({"inputs": arr, "rngs": rng1})["inputs"]
        new_arr_1_copy = augment.random_time_crop()({"inputs": arr, "rngs": rng1})[
            "inputs"
        ]
        new_arr_2 = augment.random_time_crop()({"inputs": arr, "rngs": rng2})["inputs"]

        assert jnp.allclose(new_arr_1, new_arr_1_copy)
        assert not jnp.allclose(new_arr_1, new_arr_2)
        assert new_arr_1.shape == (2, 30)


def test_time_crop():
    with settings.Settings(
        {"segment_length": 60, "cropped_length": 4, "extension": "png"}
    ):
        fn = augment.time_crop(crop_type="deterministic")
        assert isinstance(fn, Callable)

        fn = augment.time_crop(crop_type="random")
        assert isinstance(fn, Callable)

        fn = augment.time_crop(crop_type="<invalid>")
        assert fn is composition.identity


def test_rectangular_mask():
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 2)

    image_shape = (1024, 2048, 3)
    ratios = jnp.array([0.5, 0.1])

    masks, actual_ratios = augment.rectangular_mask(rngs, image_shape, ratios)

    assert masks.shape == (2, 1024, 2048, 1)
    assert jnp.allclose(masks.reshape((2, -1)).mean(1), ratios, atol=0.01)
    assert masks.min() == 0
    assert masks.max() == 1

    rngs2 = augment.batch_split(rngs, 2)[0]
    masks2, actual_ratios2 = augment.rectangular_mask(rngs2, image_shape, ratios)

    assert not jnp.allclose(masks, masks2)


def test_cutout():
    fn = augment.cutout(beta_params=None)
    assert fn == composition.identity

    fn = augment.cutout(beta_params=1.0)

    rng1 = jax.random.PRNGKey(0)
    rngs1 = jax.random.split(rng1, 1024)
    rng2 = jax.random.PRNGKey(1)
    rngs2 = jax.random.split(rng2, 1024)

    x = jnp.ones((1024, 10, 20, 3)) * 2

    v1 = fn({"inputs": x, "rngs": rngs1})
    v2 = fn({"inputs": x, "rngs": rngs2})

    assert_all_different([rngs1, rngs2, v1["rngs"], v2["rngs"]])

    x1 = v1["inputs"]
    x2 = v2["inputs"]

    assert not jnp.allclose(x1, x2)

    assert x1.shape == x2.shape == x.shape
    assert x1.min() == 0
    assert x2.min() == 0
    assert x1.max() == 2
    assert x2.max() == 2

    assert jnp.allclose(x1.mean(), 1.0, atol=0.2)
    assert jnp.allclose(x2.mean(), 1.0, atol=0.2)

    assert jnp.allclose(x1[..., 0], x1[..., 1])
    assert jnp.allclose(x1[..., 0], x1[..., 2])
    assert jnp.allclose(x2[..., 0], x2[..., 2])
    assert jnp.allclose(x2[..., 0], x2[..., 2])

    assert x1[..., 0].all(1).max(1).min() == 1
    assert x1[..., 0].all(2).max(1).min() == 1
    assert x2[..., 0].all(1).max(1).min() == 1
    assert x2[..., 0].all(2).max(1).min() == 1


def test_mixup():
    fn = augment.mixup(beta_params=None)
    assert fn == composition.identity

    fn = augment.mixup(beta_params=1.0)

    rng1 = jax.random.PRNGKey(0)
    rngs1 = jax.random.split(rng1, 1024)
    rng2 = jax.random.PRNGKey(1)
    rngs2 = jax.random.split(rng2, 1024)

    x = jnp.ones((1024, 10, 20, 3))
    x = x * jnp.arange(1024).reshape((1024, 1, 1, 1))

    lbls = jnp.arange(1024) % 5
    lbls = jax.nn.one_hot(lbls, 5)

    v1 = fn({"inputs": x, "rng": rng1, "rngs": rngs1, "one_hot_labels": lbls})
    v2 = fn({"inputs": x, "rng": rng2, "rngs": rngs2, "one_hot_labels": lbls})

    assert_all_different([rngs1, rngs2, v1["rngs"], v2["rngs"]])

    x1 = v1["inputs"]
    x2 = v2["inputs"]

    assert not jnp.allclose(x1, x2)

    assert x1.shape == x2.shape == x.shape

    assert x1.min() >= 0
    assert x2.min() >= 0
    assert not jnp.allclose(x1.min(), x2.min())
    assert x1.max() <= 1024
    assert x2.max() <= 1024
    assert not jnp.allclose(x1.max(), x2.max())

    assert jnp.allclose(x1[..., 0], x1[..., 1])
    assert jnp.allclose(x1[..., 0], x1[..., 2])
    assert jnp.allclose(x2[..., 0], x2[..., 2])
    assert jnp.allclose(x2[..., 0], x2[..., 2])

    assert x1.reshape(-1, 1).std(1).max() == 0
    assert x2.reshape(-1, 1).std(1).max() == 0

    assert jnp.allclose(x1.mean(), 511.5, atol=5)
    assert jnp.allclose(x2.mean(), 511.5, atol=5)

    lbls1 = v1["one_hot_labels"]
    lbls2 = v2["one_hot_labels"]

    assert lbls1.shape == lbls2.shape == lbls.shape
    assert jnp.allclose(lbls1.sum(1), 1)
    assert jnp.allclose(lbls2.sum(1), 1)
    assert lbls1.std(1).max() > 0
    assert lbls2.std(1).max() > 0
    assert lbls1.std(0).max() > 0
    assert lbls2.std(0).max() > 0
    assert not jnp.allclose(lbls1, lbls2)

    rng3 = jax.random.PRNGKey(2)
    rngs3 = jax.random.split(rng3, 20)
    x3 = jax.random.uniform(rng3, (20, 10, 20, 3))
    lbls3 = jnp.eye(20)

    v3 = fn({"inputs": x3, "rng": rng3, "rngs": rngs3, "one_hot_labels": lbls3})

    new_x3 = v3["inputs"]

    assert x3.shape == new_x3.shape
    assert not jnp.allclose(x3, new_x3)

    all_x3 = jnp.concatenate([x3, new_x3], 0).reshape(40, -1)
    assert jnp.linalg.matrix_rank(all_x3) == 20


def test_cutmix():
    fn = augment.cutmix(beta_params=None)
    assert fn == composition.identity

    fn = augment.cutmix(beta_params=1.0)

    rng1 = jax.random.PRNGKey(0)
    rngs1 = jax.random.split(rng1, 100)
    rng2 = jax.random.PRNGKey(1)
    rngs2 = jax.random.split(rng2, 100)

    x = jnp.ones((100, 10, 20, 3))
    x = x * jnp.arange(100).reshape((100, 1, 1, 1)) % 10

    lbls = jnp.arange(100) % 5
    lbls = jax.nn.one_hot(lbls, 5)

    v1 = fn({"inputs": x, "rng": rng1, "rngs": rngs1, "one_hot_labels": lbls})
    v2 = fn({"inputs": x, "rng": rng2, "rngs": rngs2, "one_hot_labels": lbls})

    assert_all_different([rngs1, rngs2, v1["rngs"], v2["rngs"]])

    x1 = v1["inputs"]
    x2 = v2["inputs"]

    assert not jnp.allclose(x1, x2)

    assert x1.shape == x2.shape == x.shape

    assert x1.min() == 0
    assert x2.min() == 0
    assert x1.max() == 9
    assert x2.max() == 9

    assert jnp.allclose(x1[..., 0], x1[..., 1])
    assert jnp.allclose(x1[..., 0], x1[..., 2])
    assert jnp.allclose(x2[..., 0], x2[..., 2])
    assert jnp.allclose(x2[..., 0], x2[..., 2])

    assert (x1 == x)[..., 0].all(1).max(1).min() == 1
    assert (x1 == x)[..., 0].all(2).max(1).min() == 1
    assert (x2 == x)[..., 0].all(1).max(1).min() == 1
    assert (x2 == x)[..., 0].all(2).max(1).min() == 1

    print(x1.mean(), x2.mean())
    assert jnp.allclose(x1.mean(), 4.5, atol=0.2)
    assert jnp.allclose(x2.mean(), 4.5, atol=0.2)
