import pandas as pd
import imageio
import os
import yaml
import argparse
import soundfile
import librosa
import numpy as np
import multiprocessing
from tqdm import tqdm

# disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from soundscape import settings, composition
from soundscape.composition import Composable


def parallel_map(fn, iterable):
    """
    apply a function to a list in parallel with tqdm
    """

    with multiprocessing.Pool() as pool:
        return list(tqdm(pool.imap(fn, iterable), total=len(iterable)))

    # return list(map(fn, tqdm(iterable)))


@Composable
def read_audio(values):
    """
    Read an audio file and return the audio time series and its sampling rate
    """

    filename = values["filename"]
    audio, sr = soundfile.read(filename)
    audio = audio.squeeze()
    return {**values, "audio": audio, "sr": sr}


@settings.settings_fn
def convert_to_sample_rate(*, sr):
    """
    Convert an audio time series to a different sample rate.
    """

    @Composable
    def _convert_to_sample_rate(values):
        audio = values["audio"]
        source_sr = values["sr"]

        new_audio = librosa.resample(
            audio, orig_sr=source_sr, target_sr=sr, res_type="fft"
        ).squeeze()

        return {**values, "audio": new_audio, "sr": sr}

    return _convert_to_sample_rate


@settings.settings_fn
def pad_audio_segment(*, segment_length, sr, pad_mode):
    """
    Pad an audio segment to segment_length seconds.
    """

    pad_size = int(segment_length / 2 * sr)

    @Composable
    def _pad_audio_segment(values):
        audio = values["audio"]
        audio = np.pad(audio, (pad_size, pad_size), pad_mode)
        return {**values, "audio": audio}

    return _pad_audio_segment


@settings.settings_fn
def crop_audio_segment(*, segment_length, sr):
    @Composable
    def _crop_audio_segment(values):
        audio = values["audio"]

        cropped_size = int(segment_length * sr)
        begin_idx = audio.shape[0] // 2 - cropped_size // 2
        end_idx = audio.shape[0] // 2 + cropped_size // 2

        audio = audio[begin_idx:end_idx]

        return {**values, "audio": audio}

    return _crop_audio_segment


@settings.settings_fn
def spectrogram(*, sr, spectrogram_config):
    @Composable
    def _spectrogram(values):
        audio = values["audio"]
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, **spectrogram_config)
        return {**values, "spec": spec}

    return _spectrogram


@settings.settings_fn
def process_spectrogram(*, lower_threshold, upper_threshold, precision):
    @Composable
    def _process_spectrogram(values):
        spec = values["spec"]

        spec = librosa.power_to_db(spec)

        spec = spec - spec.mean(1)[:, None]

        spec = np.clip(spec, lower_threshold, upper_threshold)
        spec = (spec - lower_threshold) / (upper_threshold - lower_threshold)

        spec = spec * (2**precision - 1)
        dtype = np.uint8 if precision == 8 else np.uint16
        spec = spec.astype(dtype)

        return {**values, "spec": spec}

    return _process_spectrogram


@settings.settings_fn
def save_file(*, data_dir, spectrogram_dir, sr):
    @Composable
    def _save_file(values):
        idx = values["id"]
        audio = values["audio"]
        spec = values["spec"]
        split = values["split"]
        label = values["label"]

        dirpath = f"{data_dir}/{spectrogram_dir}/{split}/{label}"

        os.makedirs(dirpath, exist_ok=True)

        soundfile.write(f"{dirpath}/{idx}.wav", audio, sr)
        imageio.imwrite(f"{dirpath}/{idx}.png", spec)

        return values

    return _save_file


@settings.settings_fn
def get_processing_function():
    return (
        read_audio
        | convert_to_sample_rate()
        | pad_audio_segment()
        | crop_audio_segment()
        | spectrogram()
        | process_spectrogram()
        | save_file()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("beans_path", type=str, help="Path to the dataset.yaml file")
    parser.add_argument(
        "--segment_length", type=int, help="Length of the audio segment"
    )
    parser.add_argument(
        "settings_yaml_path", type=str, help="Path to the settings yaml file"
    )

    args = parser.parse_args()

    with open(f"{args.beans_path}/datasets.yml", "r") as f:
        configs = yaml.safe_load(f)

    config = [config for config in configs if config["name"] == args.dataset_name][0]

    df_train = pd.read_csv(f"{args.beans_path}/{config['train_data']}", index_col=0)
    df_val = pd.read_csv(f"{args.beans_path}/{config['valid_data']}", index_col=0)
    df_test = pd.read_csv(f"{args.beans_path}/{config['test_data']}", index_col=0)

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_val, df_test])

    with open(args.settings_yaml_path, "r") as f:
        settings_dict = yaml.safe_load(f)

    new_settings_dict = {
        "spectrogram_dir": config["name"],
        "class_order": list(config["labels"]),
        "num_classes": len(config["labels"]),
        "labels_file": f"{config['name']}.csv",
        "segment_length": args.segment_length,
    }

    settings_dict = {**settings_dict, **new_settings_dict}

    with settings.Settings(settings_dict):
        process_fn = get_processing_function()

    with open(f'settings/{config["name"]}.yaml', "w") as f:
        yaml.dump(new_settings_dict, f)

    df.index.name = "index"
    df["class"] = df["label"]
    df["species"] = df["label"]
    df["selected"] = True
    df["exists"] = True
    df["file"] = df["path"]
    df = df.drop(columns=["path", "label"])

    df.to_csv(f"{settings_dict['data_dir']}/{config['name']}.csv")

    def _process(df_row):
        values = {
            "filename": f"{args.beans_path}/{df_row['file']}",
            "label": df_row["class"],
            "split": df_row["split"],
            "id": df_row.name,
        }

        process_fn(values)

    parallel_map(_process, list(df.iloc))
