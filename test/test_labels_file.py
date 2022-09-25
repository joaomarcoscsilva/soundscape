from glob import glob
import pandas as pd

from soundscape.data import dataset_functions as dsfn
from soundscape.lib.settings import settings


def test_missing_files():

    labels = pd.read_csv(settings["data"]["data_dir"] + "/labels.csv")
    files = glob(settings["data"]["data_dir"] + "/wavs/**/*.wav", recursive=True)
    files = {f.replace(settings["data"]["data_dir"] + "/", "") for f in files}

    existing_labels = set(labels[labels["exists"]]["file"])
    non_existing_labels = set(labels[~labels["exists"]]["file"])

    assert existing_labels == files
    assert non_existing_labels.intersection(files) == set()


def test_split_labels():
    train_labels = dsfn.get_labels("train_labels.csv")
    test_labels = dsfn.get_labels("test_labels.csv")
    val_labels = dsfn.get_labels("val_labels.csv")

    train_files = set(train_labels["filename"])
    test_files = set(test_labels["filename"])
    val_files = set(val_labels["filename"])

    # Check that there are no overlapping files (leaks)
    assert len(train_files.intersection(test_files)) == 0
    assert len(train_files.intersection(val_files)) == 0
    assert len(test_files.intersection(val_files)) == 0

    # Check that the number of files is correct
    total_files = len(train_files) + len(test_files) + len(val_files)

    test_frac = len(test_files) / total_files
    val_frac = len(val_files) / total_files

    # Checks that the proportions are correct
    assert abs(test_frac - settings["data"]["split"]["test_frac"]) < 0.01
    assert abs(val_frac - settings["data"]["split"]["val_frac"]) < 0.01
