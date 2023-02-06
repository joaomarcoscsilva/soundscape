import pandas as pd
import librosa
import os
import numpy as np
from tqdm import tqdm
import imageio
from jax import random
import multiprocessing
import soundfile

from soundscape import settings

settings.from_file()


def parallel_map(fn, iterable):
    """
    Apply a function to a list in parallel with tqdm
    """

    with multiprocessing.Pool() as pool:
        return list(tqdm(pool.imap(fn, iterable), total=len(iterable)))


@settings.settings_fn
def read_audio_file(filename, *, data_dir):
    """
    Read an audio file and return the audio time series and its sampling rate
    """

    audio, sr = librosa.load(os.path.join(data_dir, filename))
    return audio


@settings.settings_fn
def crop_audio_segment(audio, start, end, *, sr):
    """
    Crop an audio segment from start to end seconds.
    """

    start_index = int(sr * start)
    end_index = int(sr * end)
    return audio[start_index:end_index]


@settings.settings_fn
def pad_audio_segment(audio, start, end, *, segment_length, sr, pad_mode):
    """
    Pad an audio segment to segment_length seconds.
    """

    pad_length = segment_length / 2
    audio = np.pad(audio, (int(pad_length * sr), int(pad_length * sr)), pad_mode)

    start += pad_length
    end += pad_length

    return audio, start, end


@settings.settings_fn
def crop_audio_event(audio, start, end, *, segment_length, sr):
    """
    Crop an event of size segment_length from the midpoint of start and end
    """

    audio, start, end = pad_audio_segment(
        audio, start, end, segment_length=segment_length, sr=sr
    )

    begin = (start + end) / 2 - segment_length / 2
    end = begin + segment_length

    return crop_audio_segment(audio, begin, end)


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
    df = df.sort_values(["class", "file", "begin_time"])
    df = df.reset_index(drop=True)

    return df


@settings.settings_fn
def save_event(
    audio,
    spectrogram,
    split,
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

    filename = os.path.join(data_dir, spectrogram_dir, split, class_name, f"{event_id}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    soundfile.write(filename + ".wav", audio, sr)
    imageio.imwrite(filename + ".png", spectrogram)


def extract_spectrograms_from_file(df_file):
    """
    Extract the spectrograms for a given file
    """

    # Read the audio file
    audio = read_audio_file(df_file[0])

    breakpoint()

    # For each event in the file, crop the audio and compute the spectrogram
    for row in df_file[1].iloc:
        cropped_audio = crop_audio_event(audio, row["begin_time"], row["end_time"])
        spectrogram = melspectrogram(cropped_audio)
        spectrogram = process_melspectrogram(spectrogram)

        # Save the spectrogram to disk
        save_event(cropped_audio, spectrogram, row["split"], row["class"], row.name)


@settings.settings_fn
def extract_spectrograms(df):

    # Group the events belonging to the same file
    df_files = df.groupby("file")

    df_files = list(df_files)

    parallel_map(extract_spectrograms_from_file, df_files)

    return df


@settings.settings_fn
def split_class(df_class, rng, *, val_size, test_size):
    """
    Split a class into train, validation and test sets
    """

    val_samples = int(len(df_class) * val_size)
    test_samples = int(len(df_class) * test_size)
    train_samples = len(df_class) - val_samples - test_samples

    splits = np.array(
        ["train"] * train_samples + ["val"] * val_samples + ["test"] * test_samples
    )

    splits = splits[random.permutation(rng, len(splits))]

    return splits


@settings.settings_fn
def split_dataset(df, *, split_seed):

    # Group the events belonging to the same class
    df_classes = df.groupby("class")

    rng = random.PRNGKey(split_seed)

    for class_id, df_class in df_classes:
        rng, split_rng = random.split(rng)
        splits = split_class(df_class, split_rng)
        df.loc[df_class.index, "split"] = splits

    return df


if __name__ == "__main__":
    df = load_df()
    df = split_dataset(df)
    extract_spectrograms(df)
