import tensorflow as tf
import numpy as np
import pandas as pd
import os
from functools import cache
import jax.numpy as jnp
import jax

from ..lib import utils, constants
from ..lib.settings import SettingsFunction


@SettingsFunction
def class_index(settings, species):
    """
    Returns the class index of a given species.
    The returned index depends on settings["data"]["num_classes"].
    """

    if settings["data"]["num_classes"] == 13:
        return constants.CLASS_INDEX[species] if species in constants.CLASS_INDEX else 0
    elif settings["data"]["num_classes"] == 12:
        return constants.CLASS_INDEX[species] - 1
    elif settings["data"]["num_classes"] == 2:
        return 0 if species in constants.BIRD_CLASSES else 1
    else:
        raise ValueError("Invalid number of classes")


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
        return np.pad(
            x,
            [(0, constants.MAX_EVENTS - x.shape[0])] + [(0, 0)] * (x.ndim - 1),
            "constant",
            constant_values=value,
        ).astype(np.float32)

    return f


@SettingsFunction
def read_df(settings, filename):
    """
    Read the dataframe containing the labelled events.
    """

    # Reads the csv file and selects only the desired rows
    df = pd.read_csv(os.path.join(settings["data"]["data_dir"], filename))
    df = df[df["exists"]]

    if settings["data"]["num_classes"] != 13:
        df = df[df["selected"]]

    # Changes the types of some columns to float
    df.loc[:, ["begin_time", "end_time", "low_freq", "high_freq"]] = df[
        ["begin_time", "end_time", "low_freq", "high_freq"]
    ].astype(np.float32)

    # Sorts by begin_time
    df = df.sort_values(by="begin_time")

    return df


@SettingsFunction
@cache
def get_labels(settings, filename):
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

    df = read_df(filename)

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
        .apply(lambda x: x.apply(class_index).values)
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


@SettingsFunction
def num_events(settings, path="labels.csv"):
    """
    Gets the total number of events in the dataset.
    """
    return get_labels(path)["num_events"].sum()


@SettingsFunction
def class_frequencies(settings, path="labels.csv"):
    """
    Return a vector with the relative frequencies of each class.
    """

    labels = get_labels(path)["labels"].flatten()
    labels = labels[labels != -1]
    label_counts = np.bincount(labels, minlength=settings["data"]["num_classes"])
    return jnp.array(label_counts / label_counts.sum())


@SettingsFunction
def extract_waveform(settings, args):
    """
    Extract the waveform of a given audio file.
    """

    wav = tf.io.read_file(
        tf.strings.join([settings["data"]["data_dir"], args["filename"]], separator="/")
    )
    wav, sr = tf.audio.decode_wav(wav)
    wav = wav[:, 0]

    return {"wav": wav, **args}


@SettingsFunction
def load_spectrogram(settings, args):
    """
    Load the mel spectrogram from a png image.
    """

    filepath = args["filename"]
    filepath = tf.strings.regex_replace(filepath, ".wav$", ".png")
    filepath = tf.strings.regex_replace(filepath, "wavs/", "specs/")
    filepath = tf.strings.join([settings["data"]["data_dir"], filepath], separator="/")

    spec = tf.image.decode_png(tf.io.read_file(filepath), dtype=tf.uint16)
    spec = spec[:, :, 0]

    return {"spec": spec, **args}


@SettingsFunction
def extract_spectrogram(settings, args):
    """
    Extract the spectrogram of a given audio waveform.
    """

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

    args.pop("wav")

    return {"spec": mel_spectrogram, **args}


@SettingsFunction
def fragment_borders(settings, args):
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
    size using random cropping."""

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


@SettingsFunction
def prepare_batch(settings, args):
    """
    Prepare a sample for training.
    The arguments passed must be jax arrays.
    """

    args = {**args}

    if settings["data"]["downsample"] is not None:
        args["spec"] = jax.image.resize(
            args["spec"], (args["spec"].shape[0], 224, 224), method="bicubic"
        )

    args["spec"] = jnp.float32(args["spec"])
    args["spec"] = args["spec"][..., None]
    args["spec"] = args["spec"] / (256**2 - 1)
    args["spec"] = jnp.repeat(args["spec"], 3, axis=-1)

    args["labels"] = jax.nn.one_hot(args["labels"], settings["data"]["num_classes"])

    return args
