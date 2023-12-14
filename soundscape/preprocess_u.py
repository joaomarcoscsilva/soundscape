import multiprocessing
import os

import imageio
import librosa
import numpy as np
import pandas as pd
import soundfile
from jax import random
from tqdm import tqdm

from soundscape import settings

DISABLE_PARALLEL = False


def parallel_map(fn, iterable):
    """
    Apply a function to a list in parallel with tqdm
    """
    if DISABLE_PARALLEL:
        return list(tqdm(map(fn, iterable), total=len(iterable)))

    with multiprocessing.Pool() as pool:
        return list(tqdm(pool.imap(fn, iterable), total=len(iterable)))


@settings.settings_fn
def read_audio_file(filename, *, data_dir):
    """
    Read an audio file and return the audio time series and its sampling rate
    """

    audio, sr = librosa.load(os.path.join(data_dir, filename), res_type="fft")
    return audio


@settings.settings_fn
def melspectrogram(audio, *, sr, spectrogram_config):
    """
    Compute the mel spectrogram of an audio time series
    """

    return librosa.feature.melspectrogram(y=audio, sr=sr, **spectrogram_config)


@settings.settings_fn
def process_melspectrogram(spectrogram, *, precision, lower_threshold, upper_threshold):
    """
    Preprocess an audio file and return its mel spectrogram
    """

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram - spectrogram.mean(1)[:, None]

    spectrogram = np.clip(spectrogram, lower_threshold, upper_threshold)
    spectrogram = (spectrogram - lower_threshold) / (upper_threshold - lower_threshold)

    spectrogram = spectrogram * (2**precision - 1)
    dtype = np.uint8 if precision == 8 else np.uint16
    spectrogram = spectrogram.astype(dtype)

    return spectrogram


@settings.settings_fn
def load_df(*, data_dir, labels_file):
    """
    Read the labels dataframe and return it in a standard format.
    """

    df = pd.read_csv(os.path.join(data_dir, labels_file))

    # Selects only the files that exist
    df = df[df["exists"]]

    # Create the "class" column
    df["class"] = df["species"]
    df.loc[~df["selected"], "class"] = "other"

    # Sort the dataframe
    df = df.sort_values(["class", "file"])
    df = df.reset_index(drop=True)

    return df


@settings.settings_fn
def save_event(
    audio,
    spectrogram,
    class_name,
    event_id,
    *,
    data_dir,
    spectrogram_dir,
    sr,
):
    """
    Save a spectrogram to a png file
    """

    filename = os.path.join(data_dir, spectrogram_dir, class_name, f"{event_id}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # soundfile.write(filename + ".wav", audio, sr)
    imageio.imwrite(filename + ".png", spectrogram)


def extract_spectrograms_from_file(df_file):
    """
    Extract the spectrograms for a given file
    """

    # Read the audio file
    audio = read_audio_file(df_file[0])

    # For each event in the file, crop the audio and compute the spectrogram
    spectrogram = melspectrogram(audio)
    spectrogram = process_melspectrogram(spectrogram)

    # Save the spectrogram to disk
    save_event(
        audio, spectrogram, df_file[1]["class"].iloc[0], df_file[1]["index"].iloc[0]
    )


@settings.settings_fn
def extract_spectrograms(df):
    # Group the events belonging to the same file
    df_files = df.groupby("file")

    df_files = list(df_files)

    parallel_map(extract_spectrograms_from_file, df_files)

    return df


@settings.settings_fn
def split_array(array, rng, *, val_size, test_size):
    """
    Split a class into train, validation and test sets
    """

    val_samples = int(len(array) * val_size)
    test_samples = int(len(array) * test_size)
    train_samples = len(array) - val_samples - test_samples

    splits = np.array(
        ["train"] * train_samples + ["val"] * val_samples + ["test"] * test_samples
    )

    splits = splits[random.permutation(rng, len(splits))]

    return splits


if __name__ == "__main__":
    with settings.Settings.from_command_line():
        df = load_df()
        extract_spectrograms(df)
