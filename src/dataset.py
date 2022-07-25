import tensorflow as tf
import numpy as np
import pandas as pd
import os
import yaml

import constants


def stack(x):
    """
    Stacks the rows of a pandas Series
    """
    return np.stack([x[c] for c in x.index]).T


def pad(value):
    """
    Returns a function that pads the first dimension of a numpy array with the given value.

    It is padded so that the first dimension is of length `constants.MAX_EVENTS`, which
    is the maximum number of labelled events in any audio file.
    """

    def f(x):
        dims = len(x.shape)
        return np.pad(
            x,
            [(0, constants.MAX_EVENTS - x.shape[0])] + [(0, 0)] * (dims - 1),
            "constant",
            constant_values=value,
        )

    return f


def get_labels(csv_filename):
    """
    Returns the contents of the `labels.csv` file in a format usable by tf.data.Dataset.from_tensor_slices.

    Below, we have:
        - N = number of distinct labelled audio files
        - E = maximum number of events in a labelled audio file (5)

    Returns:
        - filenames: list of strings of shape (N)
        - time_intervals: tensor of shape (N, E, 2) containing the start and end times of each event
        - freq_intervals: tensor of shape (N, E, 2) containing the start and end frequencies of each event
        - labels: integer tensor of shape (N, E) containing the label of each event (out of the 12 selected classes)
    """

    # Reads the csv file and selects only the desired rows
    df = pd.read_csv(csv_filename)
    df = df[df["selected"] & df["exists"]]

    # Changes the types of some columns to float
    df.loc[:, ["begin_time", "end_time", "low_freq", "high_freq"]] = df[
        ["begin_time", "end_time", "low_freq", "high_freq"]
    ].astype(float)

    # Groups by each sound file
    df = df.groupby("file")

    # List of filenames
    filenames = list(df.groups.keys())

    # List of time intervals
    time_intervals = np.stack(
        pd.DataFrame(
            {
                "begin_time": df["begin_time"].apply(lambda x: x.values),
                "end_time": df["end_time"].apply(lambda x: x.values),
            }
        )
        .apply(stack, axis=1)
        .apply(pad(0))
        .values
    )

    # List of frequency intervals
    freq_intervals = np.stack(
        pd.DataFrame(
            {
                "low_freq": df["low_freq"].apply(lambda x: x.values),
                "high_freq": df["high_freq"].apply(lambda x: x.values),
            }
        )
        .apply(stack, axis=1)
        .apply(pad(0))
        .values
    )

    # List of labels
    labels = np.stack(
        df["species"]
        .apply(lambda x: x.apply(lambda s: constants.CLASS_INDEX[s]).values)
        .apply(pad(-1))
        .values
    )

    return (filenames, time_intervals, freq_intervals, labels)


def extract_waveform(filename, *args):
    """
    Extracts the waveform of a given audio file.
    """
    print(filename)
    wav = tf.io.read_file(filename)
    wav, sr = tf.audio.decode_wav(wav)

    return (wav, sr, *args)


def extract_melspectrogram(wav, sr, *args):
    """
    Extracts the melspectrogram of a given audio waveform.
    """
    raise NotImplementedError


def labels_dataset(labels_file):
    """
    Returns a tf.data.Dataset containing the labelled audio files.
    """
    labels = get_labels(labels_file)
    ds = tf.data.Dataset.from_tensor_slices(labels)
    return ds


def waveform_dataset(labels_file):
    """
    Returns a tf.data.Dataset containing the labelled audio files and their waveforms.
    """
    ds = labels_dataset(labels_file)
    ds = ds.map(extract_waveform)
    return ds


def melspectrogram_dataset(labels_file):
    """
    Returns a tf.data.Dataset containing the labelled audio files and their spectrograms.
    """
    ds = waveform_dataset(labels_file)
    ds = ds.map(extract_melspectrogram)
    return ds
