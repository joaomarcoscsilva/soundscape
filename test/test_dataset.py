from json import load
import tensorflow as tf
import numpy as np
import jax
from jax import numpy as jnp
import os
import pytest

from soundscape.data import dataset
from soundscape.lib import constants, utils
from soundscape.lib.settings import settings


# Number of samples to test in each dataset
n = 4


def assert_dicts_all_equal(dicts, key, value=None, dtype=None):
    if value == None:
        value = dicts[0][key]

    for d in dicts:
        assert tf.reduce_all(d[key] == value)
        if dtype is not None:
            assert d[key].dtype == dtype


@pytest.fixture
def labels_ds():
    return dataset.labels_dataset(settings)


@pytest.fixture
def waveform_ds():
    return dataset.waveform_dataset(settings)


@pytest.fixture
def spec_ds():
    return dataset.spectrogram_dataset(settings, from_disk=False)


@pytest.fixture
def loaded_spec_ds():
    return dataset.spectrogram_dataset(settings, from_disk=True)


def test_label_dataset(labels_ds, waveform_ds, spec_ds, loaded_spec_ds):

    for i, (l, w, s, ls) in enumerate(
        zip(
            labels_ds.take(n),
            waveform_ds.take(n),
            spec_ds.take(n),
            loaded_spec_ds.take(n),
        )
    ):
        assert_dicts_all_equal((l, w, s, ls), key="filename", dtype=tf.string)
        assert_dicts_all_equal((l, w, s, ls), key="time_intervals", dtype=tf.float32)
        assert_dicts_all_equal((l, w, s, ls), key="freq_intervals", dtype=tf.float32)
        assert_dicts_all_equal((l, w, s, ls), key="labels", dtype=tf.int32)
        assert_dicts_all_equal((l, w, s, ls), key="num_events", dtype=tf.int32)
        assert_dicts_all_equal((l, w, s, ls), key="index", value=tf.constant(i))

        assert_dicts_all_equal((w, s, ls), key="frag_intervals", dtype=tf.float32)

        assert tf.reduce_all(w["time_intervals"][:, 1] - w["time_intervals"][:, 0] >= 0)
        assert tf.reduce_all(w["freq_intervals"][:, 1] - w["freq_intervals"][:, 0] >= 0)

        valid_frags = w["frag_intervals"][0 : w["num_events"]]
        invalid_frags = w["frag_intervals"][w["num_events"] :]
        assert tf.reduce_all(
            valid_frags[:, 1] - valid_frags[:, 0]
            >= settings["data"]["fragmentation"]["fragment_size"]
        )
        assert tf.reduce_all(invalid_frags == 0)

        assert w["wav"].shape == (constants.SR * 60)
        assert w["wav"].dtype == tf.float32

        assert "wav" not in s
        assert "wav" not in ls

        assert s["spec"].shape == ls["spec"].shape
        assert s["spec"].dtype == ls["spec"].dtype == tf.uint16
        assert tf.reduce_all(
            tf.cast(s["spec"], tf.int32) - tf.cast(ls["spec"], tf.int32) <= 1
        )

        filename = ls["filename"].numpy().decode()
        assert filename.endswith(".wav")
        filename = filename.replace(".wav", ".png").replace("wavs/", "specs/")
        filename = os.path.join(settings["data"]["data_dir"], filename)
        loaded_file = tf.image.decode_png(tf.io.read_file(filename), dtype=tf.uint16)

        assert tf.reduce_all(
            tf.cast(s["spec"], tf.int32) - tf.cast(loaded_file[:, :, 0], tf.int32) <= 1
        )


def test_batching(loaded_spec_ds):

    for x in loaded_spec_ds.take(n):
        break

    for batch in loaded_spec_ds.take(n).batch(n // 2):
        for key in batch:
            assert batch[key].shape == (n // 2, *x[key].shape)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_add_rng(rng, loaded_spec_ds):

    ds_rng = dataset.add_rng(loaded_spec_ds, rng)
    rng = utils.jax2tf(rng).numpy()

    rngs = []

    for x, x_rng in zip(loaded_spec_ds.take(n), ds_rng.take(n)):
        assert "rng" not in x
        assert "rng" in x_rng
        assert x_rng["rng"].shape == rng.shape
        assert x_rng["rng"].dtype == rng.dtype
        assert not tf.reduce_all(x_rng["rng"] == rng)
        rngs.append(str(x_rng["rng"].numpy()))

    assert len(set(rngs)) == len(rngs)


def test_fragment_dataset(rng, loaded_spec_ds):

    fragment_ds = dataset.fragment_dataset(settings, loaded_spec_ds, rng)

    entries = [x for x in loaded_spec_ds.take(n)]

    fragment_ds_it = fragment_ds.as_numpy_iterator()

    for entry in entries:
        for i in range(entry["num_events"].numpy()):
            x = next(fragment_ds_it)

            for key in ["filename", "num_events"]:
                assert x[key] == entry[key].numpy()

            for key in ["frag_intervals", "freq_intervals", "time_intervals"]:
                assert tf.reduce_all(x[key] == entry[key][i])

            assert x["spec"].ndim == 2
            assert x["spec"].shape[-1] == settings["data"]["spectrogram"]["n_mels"]


def test_jax_dataset(loaded_spec_ds):
    ds = loaded_spec_ds.take(n)
    jaxds = dataset.jax_dataset(ds)

    for x_tf, x_jax in zip(ds, jaxds):
        for key in x_tf:
            if x_tf[key].dtype == tf.string:
                assert x_tf[key].numpy().decode() == x_jax[key]
            else:
                assert jnp.all(x_tf[key].numpy() == x_jax[key])
