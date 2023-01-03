import pandas as pd
import librosa
import os
import numpy as np
import settings
from tqdm import tqdm
import imageio

settings, settings_dict = settings.from_file("preprocess_config.yaml")


@settings
def read_audio_file(filename, *, data_dir):
    """
    Read an audio file and return the audio time series and its sampling rate
    """

    print(data_dir)

    audio, sr = librosa.load(os.path.join(data_dir, filename))
    return audio


@settings
def crop_audio_segment(audio, start, end, *, sr):
    """
    Crop an audio segment from start to end seconds.
    """

    start_index = int(sr * start)
    end_index = int(sr * end)
    return audio[start_index:end_index]


@settings
def crop_audio_event(audio, start, end, *, segment_length):
    """
    Crop an event of size segment_length from the midpoint of start and end
    """

    begin = (start + end) / 2 - segment_length / 2
    begin = max(begin, 0)
    end = begin + segment_length

    return crop_audio_segment(audio, begin, end)


@settings
def melspectrogram(audio, *, sr, spectrogram_config):
    """
    Compute the mel spectrogram of an audio time series
    """

    return librosa.feature.melspectrogram(y=audio, sr=sr, **spectrogram_config)


@settings
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


@settings
def load_df(*, data_dir, labels_file, class_order):
    df = pd.read_csv(os.path.join(data_dir, labels_file))

    # Selects only the files that exist
    df = df[df["exists"]]

    # Create the "class" column
    df["class"] = df["species"]
    df.loc[~df["selected"], "class"] = "other"
    df["class"] = df["class"].map(class_order.index)

    # Sort the dataframe
    df = df.sort_values(["class", "file", "begin_time"])
    df = df.reset_index(drop=True)

    return df


@settings
def save_melspectrogram(
    spectrogram, *, class_id, id, data_dir, spectrogram_dir, class_order, precision
):
    """
    Save a spectrogram to a png file
    """

    filename = os.path.join(
        data_dir, spectrogram_dir, class_order[class_id], f"{id}.png"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # imageio.imwrite(filename, spectrogram)


@settings
def main():
    df = load_df()
    df_files = df.groupby("file")

    def produce_melspectrogram(audio_file):
        audio = read_audio_file(audio_file)

        for row in df_files.get_group(audio_file).iloc:
            cropped_audio = crop_audio_event(audio, row["begin_time"], row["end_time"])
            spectrogram = melspectrogram(cropped_audio)
            spectrogram = process_melspectrogram(spectrogram)

            save_melspectrogram(spectrogram, row["class"], row.name)

    for file in tqdm(list(df["file"].unique())):
        produce_melspectrogram(file)

    return df


if __name__ == "__main__":
    main()
