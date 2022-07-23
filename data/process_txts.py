from functools import partial
import pandas as pd
import sys, os
from tqdm import tqdm

import constants


def class_is_frog(species):
    return species in constants.FROG_CLASSES


def class_is_bird(species):
    return species in constants.BIRD_CLASSES


def process_txt_file(directory, period, filename):
    """
    Takes a path to a .txt labels file and returns a pandas dataframe.
    """

    # Reads the .txt file
    df = pd.read_csv(os.path.join(directory, period, filename), sep="\t")

    # Drops the columns that are not needed
    df = df.drop(columns=["Selection", "View", "Channel"])

    df["file"] = filename.replace(".txt", ".wav")
    df["period"] = period

    # Creates series to indicate whether a species is part of the pre-selected 12 classes (6 birds, 6 frogs)
    isBird = df["species"].apply(class_is_bird)
    isFrog = df["species"].apply(class_is_frog)

    # Creates a column to indicate whether a species is part of the pre-selected classes
    df["selected"] = isBird | isFrog

    # Indicates whether a species is a bird or a frog
    df["group"] = pd.Series(dtype=str)
    df.loc[isBird, "group"] = "bird"
    df.loc[isFrog, "group"] = "frog"

    return df


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python process_txts.py <directory>")
        sys.exit(1)

    path = sys.argv[1]

    periods = os.listdir(path)

    dfs = []

    for period in periods:
        files = os.listdir(os.path.join(path, period))
        dfs.extend(map(partial(process_txt_file, path, period), tqdm(files)))

    df = pd.concat(dfs)

    df.to_csv("labels.csv", index=False)
