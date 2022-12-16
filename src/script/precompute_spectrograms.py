from tqdm.auto import tqdm
import tensorflow as tf
import imageio
import os

from ..lib.settings import settings
from ..data import dataset


if __name__ == "__main__":

    ds = dataset.spectrogram_dataset(from_disk=False)

    for d in tqdm(ds):

        # Finds the name of the image to be saved
        file = (
            d["file"].numpy().decode().replace(".wav", ".png").replace("wavs/", "")
        )
        file = os.path.join(settings["data"]["data_dir"], "specs", file)

        # Creates the necessary directories if they don't exist
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))

        # Saves the image
        imageio.imwrite(file, d["spec"].numpy())
