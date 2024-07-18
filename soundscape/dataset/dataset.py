import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import imageio
import librosa
import pandas as pd
import soundfile
import tensorflow as tf
import tensorflow_io as tfio
from omegaconf import DictConfig
from p_tqdm import p_map

from . import preprocessing, split


def _read_audio_file(path: str, sr: int):
    audio = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)[0]
    if sr != 22050:
        audio = tfio.audio.resample(audio, 22050, sr)
    return audio


def _read_image_file(path: str, image_precision: int):
    dtype = tf.uint16 if image_precision == 16 else tf.uint8
    return tf.io.decode_png(tf.io.read_file(path), channels=1, dtype=dtype)


def load_labels_dataframe(labels_file: str) -> pd.DataFrame:
    """
    Read the labels dataframe and return it in a standard format.
    """

    df = pd.read_csv(labels_file)
    df = df[df["exists"]]

    df["class"] = df["species"]
    df.loc[~df["selected"], "class"] = "other"

    df = df.sort_values(["class", "file", "begin_time"])
    df = df.reset_index(drop=True)

    return df


@dataclass
class Dataset:
    """
    Represent an entire dataset. For training, use the DataLoader class.

    The preprocessing function below should only be called once per dataset,
    since its results are saved in the specified dataset dir.
    """

    dataset_dir: str = None
    data_type: str = None
    sample_length: float = None
    sr: int = None
    source_dir: str = None

    splitting: DictConfig = None
    preprocessing: DictConfig = None

    def preprocess(self):
        df = load_labels_dataframe(self.preprocessing.labels_file)
        df = split.split_dataframe(df, **self.splitting)
        df_files = list(df.groupby("file"))
        p_map(self._process_file, df_files)

    def reading_function(self) -> Callable[[str], tf.Tensor]:
        if self.data_type == "audio":
            return partial(_read_audio_file, sr=self.sr)

        elif self.data_type == "image":
            return partial(
                _read_image_file, image_precision=self.preprocessing.image_precision
            )

        else:
            raise ValueError(f"Unknown data type: {self.data_type}")

    def _process_file(self, df_file):
        """
        Extract the spectrograms for a given file
        """

        audio, sr = librosa.load(self.source_dir + "/" + df_file[0], res_type="fft")

        if sr != self.sr:
            audio = preprocessing.convert_to_sample_rate(audio, sr, self.sr)

        for row in df_file[1].iloc:
            cropped_audio = preprocessing.crop_centered_segment(
                audio,
                row["begin_time"],
                row["end_time"],
                sr,
                self.sample_length,
                self.preprocessing.pad_mode,
            )

            output_filename = os.path.join(
                self.dataset_dir, row["split"], row["class"], str(row.name)
            )
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            if self.data_type == "image":
                spectrogram = preprocessing.generate_melspectrogram(
                    cropped_audio, sr, self.preprocessing.spectrogram_kwargs
                )

                spectrogram = preprocessing.process_melspectrogram(
                    spectrogram,
                    self.preprocessing.image_precision,
                    self.preprocessing.thresholds,
                )

                imageio.imwrite(output_filename + ".png", spectrogram)

            else:
                soundfile.write(output_filename + ".wav", cropped_audio, sr)
