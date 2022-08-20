from this import d
import tensorflow as tf
import numpy as np
from jax import numpy as jnp
import pandas as pd
import pytest
from functools import partial

import dataset_functions as dsfn
import constants
import utils

settings = utils.hash_dict(
    {
        "data_dir": "data",
        "window_size": 2048,
        "hop_size": 256,
        "n_fft": 2048,
        "window_fn": "hamming_window",
        "n_mels": 256,
        "min_overlap": 0.5,
        "fragment_size": 5,
        "padding_mode": "edge",
        "begin_time_fn": "uniform",
    }
)


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
    df = dsfn.read_df(settings)
    assert df["exists"].all()
    assert df["begin_time"].dtype == np.float32
    assert df["end_time"].dtype == np.float32
    assert df["low_freq"].dtype == np.float32
    assert df["high_freq"].dtype == np.float32
    assert df["file"].unique().shape[0] == 4260

    assert len(df) == 16636


test_indexes = [0, 1, 1000, 4260 - 1]


@pytest.fixture
def labels():
    return dsfn.get_labels(settings)

@pytest.fixture
def wavs(labels):
    fn = dsfn.get_extract_waveform_fn(settings)
    
    def index_labels(i):
        return {k: v[i] for k, v in labels.items()}

    return list(map(fn, map(index_labels, test_indexes)))

@pytest.fixture
def specs(wavs):
    fn = dsfn.get_extract_melspectrogram_fn(settings)
    return list(map(fn, wavs))

@pytest.fixture 
def loaded_specs(labels):
    fn = dsfn.get_load_melspectrogram_fn(settings)

    def index_labels(i):
        return {k: v[i] for k, v in labels.items()}

    return list(map(fn, map(index_labels, test_indexes)))

@pytest.fixture
def frags(specs):
    fn = dsfn.get_fragment_borders_fn(settings)
    return list(map(fn, specs))


def test_get_labels(labels):

    assert len(np.unique(labels["filename"])) == 4260
    assert len(labels["filename"]) == 4260

    assert labels["time_intervals"].shape == (4260, constants.MAX_EVENTS, 2)
    assert labels["time_intervals"].dtype == np.float32

    assert labels["freq_intervals"].shape == (4260, constants.MAX_EVENTS, 2)
    assert labels["freq_intervals"].dtype == np.float32

    assert labels["labels"].shape == (4260, constants.MAX_EVENTS)
    assert labels["labels"].dtype == np.int32

    assert labels["labels"].min() == -1  # 12 selected classes + 1 for other classes
    assert labels["labels"].max() == 12


def test_extract_waveform(wavs):

    for wav in wavs:
        assert wav["wav"].shape == (60 * constants.SR,)
        assert wav["wav"].dtype == np.float32
        assert tf.math.reduce_min(wav["wav"]) >= -1
        assert tf.math.reduce_max(wav["wav"]) <= 1


def test_extract_melspectrogram(specs):

    for spec in specs:
        assert spec["spec"].shape == (settings["n_mels"], 10328)
        assert spec["spec"].dtype == np.uint16
        assert tf.math.reduce_min(spec["spec"]) >= 0


def test_load_mel_spectrogram(specs, loaded_specs):
    for spec, loaded_spec in zip(specs, loaded_specs):
        for k in loaded_spec.keys():
            if k != 'spec':
                assert np.all(spec[k] == loaded_spec[k])
            else:
                # checks that the saved spectrogram is wrong by a difference of at most 1
                assert tf.reduce_max(tf.abs(tf.cast(spec[k], tf.int32) - tf.cast(loaded_spec[k], tf.int32))) <= 1
        
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
                # Checks that the size of the fragment is at least settings["fragment_size"]
                assert interval[i, 1] - interval[i, 0] > settings["fragment_size"]

                # Checks that the fragment contains a labelled event
                assert interval[i, 0] < frag["time_intervals"][i, 0]
                assert interval[i, 1] > frag["time_intervals"][i, 1]

        