import tensorflow as tf
import numpy as np
from jax import numpy as jnp
import pandas as pd
import pytest
from functools import partial

from soundscape.data import dataset_functions as dsfn
from soundscape.lib import constants
from soundscape.lib.settings import settings

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


NUM_SAMPLES = 4260 if settings["data"]["num_classes"] == 13 else 2356
NUM_FRAGMENTS = 16636 if settings["data"]["num_classes"] == 13 else 3809


@pytest.mark.parametrize(
    "num_classes,species,expected",
    [
        (13, "[INVALID_CLASS]", 0),
        (13, "basi_culi", 1),
        (12, "dend_minu", 6),
        (2, "dend_minu", 1),
    ],
)
def test_class_index(num_classes, species, expected):
    local_settings = settings.copy()
    local_settings["data"]["num_classes"] = num_classes
    assert dsfn.class_index.call(local_settings, species) == expected


def test_fail_class_index():
    local_settings = settings.copy()
    local_settings["data"]["num_classes"] = 14  # invalid number of classes

    with pytest.raises(ValueError):
        dsfn.class_index.call(local_settings, "invalid_class")


@pytest.mark.parametrize(
    "x,y",
    [
        (pd.Series([[1], [2], [3]]), np.array([[1, 2, 3]])),
        (pd.Series([[1, 2, 3]]), np.array([[1], [2], [3]])),
    ],
)
def test_stack(x, y):
    assert (dsfn.stack(x) == y).all()


@pytest.mark.parametrize(
    "value,x,y",
    [
        (1, np.zeros(10), np.array([0] * 10 + [1] * (constants.MAX_EVENTS - 10))),
        (-1, np.zeros(10), np.array([0] * 10 + [-1] * (constants.MAX_EVENTS - 10))),
        (0, np.ones(2), np.array([1] * 2 + [0] * (constants.MAX_EVENTS - 2))),
        (1, np.zeros(0), np.ones(constants.MAX_EVENTS)),
        (1, np.zeros(constants.MAX_EVENTS), np.zeros(constants.MAX_EVENTS)),
    ],
)
def test_pad(value, x, y):
    return (dsfn.get_pad_fn(value)(x) == y).all()


def test_read_df():
    df = dsfn.read_df.call(settings, "labels.csv")

    assert df["exists"].all()
    assert df["begin_time"].dtype == np.float32
    assert df["end_time"].dtype == np.float32
    assert df["low_freq"].dtype == np.float32
    assert df["high_freq"].dtype == np.float32
    assert df["file"].unique().shape[0] == NUM_SAMPLES

    assert len(df) == NUM_FRAGMENTS


test_indexes = [0, 1, 1000, NUM_SAMPLES - 1]


@pytest.fixture
def labels():
    return dsfn.get_labels("labels.csv")


@pytest.fixture
def wavs(labels):
    def index_labels(i):
        return {k: v[i] for k, v in labels.items()}

    return list(map(dsfn.extract_waveform, map(index_labels, test_indexes)))


@pytest.fixture
def specs(wavs):
    return list(map(dsfn.extract_spectrogram, wavs))


@pytest.fixture
def loaded_specs(labels):
    def index_labels(i):
        return {k: v[i] for k, v in labels.items()}

    return list(map(dsfn.load_spectrogram, map(index_labels, test_indexes)))


@pytest.fixture
def frags(specs):
    return list(map(dsfn.fragment_borders, specs))


def test_get_labels(labels):

    assert len(np.unique(labels["filename"])) == NUM_SAMPLES
    assert len(labels["filename"]) == NUM_SAMPLES

    assert labels["time_intervals"].shape == (NUM_SAMPLES, constants.MAX_EVENTS, 2)
    assert labels["time_intervals"].dtype == np.float32

    assert labels["freq_intervals"].shape == (NUM_SAMPLES, constants.MAX_EVENTS, 2)
    assert labels["freq_intervals"].dtype == np.float32

    assert labels["labels"].shape == (NUM_SAMPLES, constants.MAX_EVENTS)
    assert labels["labels"].dtype == np.int32

    assert labels["labels"].min() == -1  # 12 selected classes + 1 for other classes
    assert labels["labels"].max() == settings["data"]["num_classes"] - 1


def test_num_events():
    assert dsfn.num_events() == NUM_FRAGMENTS


def test_class_frequencies(labels):
    freqs = dsfn.class_frequencies("labels.csv")
    assert freqs.shape == (settings["data"]["num_classes"],)
    assert freqs.dtype == np.float32
    assert freqs.sum() == 1.0
    assert freqs.min() >= 0.0
    assert freqs.max() <= 1.0


def test_extract_waveform(wavs):

    for wav in wavs:
        assert wav["wav"].shape == (60 * constants.SR,)
        assert wav["wav"].dtype == np.float32
        assert tf.math.reduce_min(wav["wav"]) >= -1
        assert tf.math.reduce_max(wav["wav"]) <= 1


def test_extract_spectrogram(specs):

    for spec in specs:
        assert spec["spec"].shape == (10328, settings["data"]["spectrogram"]["n_mels"])
        assert spec["spec"].dtype == np.uint16
        assert tf.math.reduce_min(spec["spec"]) >= 0


def test_load_mel_spectrogram(specs, loaded_specs):
    for spec, loaded_spec in zip(specs, loaded_specs):
        for k in loaded_spec.keys():
            if k != "spec":
                assert np.all(spec[k] == loaded_spec[k])
            else:
                # checks that the saved spectrogram is wrong by a difference of at most 1
                assert (
                    tf.reduce_max(
                        tf.abs(
                            tf.cast(spec[k], tf.int32)
                            - tf.cast(loaded_spec[k], tf.int32)
                        )
                    )
                    <= 1
                )


def test_fragment_borders(frags):
    for frag in frags:
        interval = frag["frag_intervals"]

        assert interval.shape == (constants.MAX_EVENTS, 2)

        # Repeats for each fragment
        for i in range(constants.MAX_EVENTS):

            # If it's an invalid fragment, the interval must be [0,0]
            if frag["labels"][i] == -1:
                assert interval[i, 0] == interval[i, 1] == 0

            else:
                # Checks that the size of the fragment is at least settings["data"]["fragmentation"]["fragment_size"]
                assert (
                    interval[i, 1] - interval[i, 0]
                    > settings["data"]["fragmentation"]["fragment_size"]
                )

                # Checks that the fragment contains a labelled event
                assert interval[i, 0] < frag["time_intervals"][i, 0]
                assert interval[i, 1] > frag["time_intervals"][i, 1]


def test_prepare_batch():
    args = {
        "spec": jnp.ones((2, 860, 256)) * (256 * 256 - 1),
        "labels": jnp.array([0, 1]),
    }

    local_settings = settings.copy()
    local_settings["data"]["downsample"] = True
    local_settings["data"]["num_classes"] = 2

    batch = dsfn.prepare_batch.call(local_settings, args)

    assert jnp.abs(batch["spec"] - jnp.ones((2, 224, 224, 3))).max() < 1e-4
    assert batch["spec"].dtype == jnp.float32
    assert jnp.all(batch["labels"] == jnp.array([[1, 0], [0, 1]]))
