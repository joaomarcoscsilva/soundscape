import tensorflow as tf
import numpy as np
import pandas as pd
import os

import constants
import utils


def stack(x):
    """
    Stacks the rows of a pandas Series
    """
    return np.stack([x[c] for c in x.index]).T


def pad_fn(value):
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
        ).astype(np.float32)

    return f


def read_df(settings):
    """
    Reads the dataframe containing the labelled events.
    """

    # Reads the csv file and selects only the desired rows
    df = pd.read_csv(os.path.join(settings["data_dir"], "labels.csv"))
    df = df[df["exists"]]

    # Changes the types of some columns to float
    df.loc[:, ["begin_time", "end_time", "low_freq", "high_freq"]] = df[
        ["begin_time", "end_time", "low_freq", "high_freq"]
    ].astype(float)

    # Sorts by begin_time
    df = df.sort_values(by="begin_time")

    return df


def get_labels(settings):
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

    df = read_df(settings)

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
        .apply(pad_fn(0))
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
        .apply(pad_fn(0))
        .values
    )

    # List of labels
    labels = np.stack(
        df["species"]
        .apply(
            lambda x: x.apply(
                lambda s: constants.CLASS_INDEX[s] if s in constants.CLASS_INDEX else 0
            ).values
        )
        .apply(pad_fn(-1))
        .values
    )

    return {
        "filename": filenames,
        "time_intervals": time_intervals,
        "freq_intervals": freq_intervals,
        "labels": labels,
    }


def extract_waveform_fn(settings):
    """
    Extracts the waveform of a given audio file.
    """

    def f(args):
        wav = tf.io.read_file(
            tf.strings.join([settings["data_dir"], args["filename"]], separator="/")
        )
        wav, sr = tf.audio.decode_wav(wav)
        wav = wav[:, 0]

        return {"wav": wav, **args}

    return f


def extract_melspectrogram_fn(settings):
    """
    Extracts the melspectrogram of a given audio waveform.
    """

    def extract_melspectrogram(args):
        stft = tf.signal.stft(
            args["wav"],
            frame_length=settings["window_size"],
            frame_step=settings["hop_size"],
            fft_length=settings["n_fft"],
            pad_end=False,
            window_fn=getattr(tf.signal, settings["window_fn"]),
        )

        mag = tf.abs(stft)

        mel = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=settings["n_mels"],
            num_spectrogram_bins=mag.shape[-1],
            sample_rate=constants.SR,
            lower_edge_hertz=10,
            upper_edge_hertz=constants.SR // 2,
            dtype=tf.float32,
        )

        mel_spectrogram = tf.matmul(mag, mel)

        return {"spec": mel_spectrogram, **args}

    return extract_melspectrogram


def fragment_borders_fn(settings):
    """
    To define the fragments used for training, we sample
    random intervals of length `settings["fragment_size"]`
    and discard those that don't sufficiently overlap a
    labelled event according to `settings["min_overlap"]`.

    Since this will be called inside a `tf.data.Dataset.map`
    call, we can't use the `jax.random` module since it's
    not possible to pass an rng argument. While we could use
    `tf.random` instead, it wouldn't have the same reproducible
    behavior as jax.

    Therefore, instead of returning a single sample of size
    `settings["fragment_size"]`, this function returns one
    larger fragment for each labelled event, such that the
    returned fragments contain every valid fragment according
    to the minimum overlap requirement. This can then be at
    train time to randomly sample a fragment of the desired
    size using random cropping.

    TODO: figure out if there will be some bias, since we'll
    sample one segment per labelled event instead of
    proportionally to the duration of the events.
    """

    def fragment_borders(args):

        border_sizes = settings["fragment_size"] * (1 - settings["min_overlap"])

        begin_times = args["time_intervals"][:, 0] - border_sizes
        end_times = args["time_intervals"][:, 1] + border_sizes

        begin_times = tf.clip_by_value(begin_times, 0, 60.0 - settings["fragment_size"])
        end_times = tf.clip_by_value(end_times, settings["fragment_size"], 60.0)

        frag_intervals = tf.stack([begin_times, end_times], axis=1)

        return {"frag_intervals": frag_intervals, **args}

    return fragment_borders
