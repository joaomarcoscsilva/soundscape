from tqdm.auto import tqdm
import tensorflow as tf
import imageio
import os

import utils
import dataset

settings = utils.hash_dict(
    {
        "data_dir": "data",
        "window_size": 2048,
        "hop_size": 256,
        "n_fft": 2048,
        "window_fn": "hamming_window",
        "n_mels": 256,
        "min_overlap": 0.5,
        "fragment_size": 5,
        "padding_mode": "edge",
        "begin_time_fn": "uniform",
    }
)

if __name__ == "__main__":

    ds = dataset.melspectrogram_dataset(settings)

    for d in tqdm(ds):

        # Finds the name of the image to be saved
        filename = d["filename"].numpy().decode().replace(".wav", ".png")
        filename = os.path.join(settings["data_dir"], "specs", filename)

        # Creates the necessary directories if they don't exist
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        # Saves the image
        imageio.imwrite(filename, d["spec"].numpy())
