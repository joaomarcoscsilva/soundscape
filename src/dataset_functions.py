import tensorflow as tf
import numpy as np
import pandas as pd
import os
from functools import cache

import constants
import utils


def stack(x):
    """
    Stack the rows of a pandas Series
    """
    return np.stack([x[c] for c in x.index]).T


def get_pad_fn(value):
    """
    Return a function that pads the first dimension of a numpy array with the given value.

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
    Read the dataframe containing the labelled events.
    """

    # Reads the csv file and selects only the desired rows
    df = pd.read_csv(os.path.join(settings["data"]["data_dir"], "labels.csv"))
    df = df[df["exists"]]

    # Changes the types of some columns to float
    df.loc[:, ["begin_time", "end_time", "low_freq", "high_freq"]] = df[
        ["begin_time", "end_time", "low_freq", "high_freq"]
    ].astype(np.float32)

    # Sorts by begin_time
    df = df.sort_values(by="begin_time")

    return df


@cache
def get_labels(settings):
    """
    Return the contents of the `labels.csv` file in a format usable by tf.data.Dataset.from_tensor_slices.

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
        .apply(get_pad_fn(0))
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
        .apply(get_pad_fn(0))
        .values
    )

    # Number of valid events for each audio file
    num_events = df["species"].size().values.astype(np.int32)

    # List of labels
    labels = np.stack(
        df["species"]
        .apply(
            lambda x: x.apply(
                lambda s: constants.CLASS_INDEX[s] if s in constants.CLASS_INDEX else 0
            ).values
        )
        .apply(get_pad_fn(-1))
        .values
    ).astype(np.int32)

    indexes = np.arange(len(filenames), dtype=np.int32)

    return {
        "filename": filenames,
        "time_intervals": time_intervals,
        "freq_intervals": freq_intervals,
        "labels": labels,
        "num_events": num_events,
        "index": indexes,
    }


def get_extract_waveform_fn(settings):
    """
    Extract the waveform of a given audio file.
    """

    def f(args):
        wav = tf.io.read_file(
            tf.strings.join(
                [settings["data"]["data_dir"], args["filename"]], separator="/"
            )
        )
        wav, sr = tf.audio.decode_wav(wav)
        wav = wav[:, 0]

        return {"wav": wav, **args}

    return f


def get_load_melspectrogram_fn(settings):
    """
    Load the mel spectrogram from a png image.
    """

    def f(args):
        filepath = args["filename"]
        filepath = tf.strings.regex_replace(filepath, ".wav$", ".png")
        filepath = tf.strings.regex_replace(filepath, "wavs/", "specs/")
        filepath = tf.strings.join(
            [settings["data"]["data_dir"], filepath], separator="/"
        )

        spec = tf.image.decode_png(tf.io.read_file(filepath), dtype=tf.uint16)
        spec = spec[:, :, 0]

        return {"spec": spec, **args}

    return f


def get_extract_melspectrogram_fn(settings):
    """
    Extract the melspectrogram of a given audio waveform.
    """

    def extract_melspectrogram(args):
        stft = tf.signal.stft(
            args["wav"],
            frame_length=settings["data"]["spectrogram"]["window_size"],
            frame_step=settings["data"]["spectrogram"]["hop_size"],
            fft_length=settings["data"]["spectrogram"]["n_fft"],
            pad_end=False,
            window_fn=getattr(tf.signal, settings["data"]["spectrogram"]["window_fn"]),
        )

        mag = tf.abs(stft)

        mel = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=settings["data"]["spectrogram"]["n_mels"],
            num_spectrogram_bins=mag.shape[-1],
            sample_rate=constants.SR,
            lower_edge_hertz=10,
            upper_edge_hertz=constants.SR // 2,
            dtype=tf.float32,
        )

        mel_spectrogram = tf.matmul(mag, mel)

        mel_spectrogram = tf.math.log(mel_spectrogram + 1)
        mel_spectrogram = mel_spectrogram - tf.reduce_mean(
            mel_spectrogram, axis=1, keepdims=True
        )
        mel_spectrogram = tf.clip_by_value(mel_spectrogram, 0, 6.5)
        mel_spectrogram = mel_spectrogram / 6.5 * 256**2
        mel_spectrogram = tf.cast(mel_spectrogram, tf.uint16)
        mel_spectrogram = tf.transpose(mel_spectrogram)

        args.pop("wav")

        return {"spec": mel_spectrogram, **args}

    return extract_melspectrogram


def get_fragment_borders_fn(settings):
    """
    To define the fragments used for training, we sample
    random intervals of length `settings["data"]["fragmentation"]["fragment_size"]`
    and discard those that don't sufficiently overlap a
    labelled event according to `settings["data"]["fragmentation"]["min_overlap"]`.

    Since this will be called inside a `tf.data.Dataset.map`
    call, we can't use the `jax.random` module since it's
    not possible to pass an rng argument. While we could use
    `tf.random` instead, it wouldn't have the same reproducible
    behavior as jax.

    Therefore, instead of returning a single sample of size
    `settings["data"]["fragmentation"]["fragment_size"]`, this
    function returns one larger fragment for each labelled event,
    such that the returned fragments contain every valid fragment
    according to the minimum overlap requirement. This can then be
    used at train time to randomly sample a fragment of the desired
    size using random cropping.
    """

    def fragment_borders(args):

        border_sizes = settings["data"]["fragmentation"]["fragment_size"] * (
            1 - settings["data"]["fragmentation"]["min_overlap"]
        )

        is_valid_event = args["labels"] != -1

        begin_times = tf.where(
            is_valid_event,
            args["time_intervals"][:, 0] - border_sizes,
            tf.zeros_like(args["time_intervals"][:, 0]),
        )

        end_times = tf.where(
            is_valid_event,
            args["time_intervals"][:, 1] + border_sizes,
            tf.zeros_like(args["time_intervals"][:, 1]),
        )

        frag_intervals = tf.stack([begin_times, end_times], axis=1)

        return {"frag_intervals": frag_intervals, **args}

    return fragment_borders
