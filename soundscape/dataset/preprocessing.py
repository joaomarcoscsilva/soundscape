from functools import partial

import jax
import librosa
import numpy as np

from . import augment, data_utils


def convert_to_sample_rate(audio, source_sr, target_sr):
    """
    Convert an audio time series to a different sample rate.
    """

    new_audio = librosa.resample(
        audio, orig_sr=source_sr, target_sr=target_sr, res_type="fft"
    ).squeeze()

    return new_audio


def crop_segment(audio, start: float, end: float, sr: int):
    """
    Crop an audio segment from start to end seconds.
    """

    start_index = int(sr * start)
    end_index = int(sr * end)
    return audio[start_index:end_index]


def pad_segment(
    audio,
    start: float,
    end: float,
    sr: int,
    sample_length: float,
    pad_mode: str = "reflect",
):
    """
    Pad an audio segment to the specified amount of seconds.
    """

    pad_length = sample_length / 2
    audio = np.pad(audio, (int(pad_length * sr), int(pad_length * sr)), pad_mode)

    start += pad_length
    end += pad_length

    return audio, start, end


def crop_centered_segment(
    audio,
    start: float,
    end: float,
    sr: int,
    sample_length: float,
    pad_mode: str = "reflect",
):
    """
    Crop an event of a specified size from the midpoint of start and end
    """

    audio, start, end = pad_segment(audio, start, end, sr, sample_length, pad_mode)

    begin = (start + end) / 2 - sample_length / 2
    end = begin + sample_length

    return crop_segment(audio, begin, end, sr)


def generate_melspectrogram(audio, sr: int, spectrogram_kwargs: dict):
    """
    Compute the mel spectrogram of an audio time series
    """

    return librosa.feature.melspectrogram(y=audio, sr=sr, **spectrogram_kwargs)


def process_melspectrogram(spectrogram, precision: int, thresholds: list[float]):
    """
    Preprocess an audio file and return its mel spectrogram
    """

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram - spectrogram.mean(1)[:, None]

    spectrogram = np.clip(spectrogram, thresholds[0], thresholds[1])
    spectrogram = (spectrogram - thresholds[0]) / (thresholds[1] - thresholds[0])

    spectrogram = spectrogram * (2**precision - 1)
    dtype = np.uint8 if precision == 8 else np.uint16
    spectrogram = spectrogram.astype(dtype)

    return spectrogram


@partial(jax.jit, static_argnames=["training", "dataloader", "augs_config"])
def preprocess(batch, training: bool, *, dataloader, augs_config):
    """
    Preprocess a batch of training data.
    """

    crop_type = augs_config.crop_type if training else "center"

    batch = data_utils.prepare_images(batch)
    batch = data_utils.one_hot_encode(batch, dataloader.num_classes)

    batch = augment.crop_inputs(
        batch, crop_type, dataloader.dataset.sample_length, augs_config.cropped_length
    )

    batch = data_utils.downsample_images(batch)

    if training:
        batch = augment.cutmix(batch, augs_config.cutmix_alpha)
        batch = augment.mixup(batch, augs_config.mixup_alpha)

    return batch
