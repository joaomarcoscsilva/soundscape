from glob import glob
import pandas as pd

from soundscape.data import dataset_functions as dsfn
from soundscape.lib.settings import settings

import pytest


@pytest.fixture
def wav_files():
    files = glob(settings["data"]["data_dir"] + "/wavs/**/*.wav", recursive=True)
    files = {f.replace(settings["data"]["data_dir"] + "/", "") for f in files}
    return files


@pytest.mark.parametrize(
    "filename", ["labels.csv", "train_labels.csv", "test_labels.csv", "val_labels.csv"]
)
def test_missing_files(wav_files, filename):

    labels = pd.read_csv(settings["data"]["data_dir"] + "/" + filename)

    existing_labels = set(labels[labels["exists"]]["file"])
    non_existing_labels = set(labels[~labels["exists"]]["file"])

    assert existing_labels.intersection(wav_files) == existing_labels
    assert non_existing_labels.intersection(wav_files) == set()


def test_split_labels():

    all_labels = dsfn.get_labels(settings)("labels.csv")

    train_labels = dsfn.get_labels(settings)("train_labels.csv")
    test_labels = dsfn.get_labels(settings)("test_labels.csv")
    val_labels = dsfn.get_labels(settings)("val_labels.csv")

    all_files = set(all_labels["filename"])
    train_files = set(train_labels["filename"])
    test_files = set(test_labels["filename"])
    val_files = set(val_labels["filename"])

    # Check that there are no overlapping files (leaks)
    assert len(train_files.intersection(test_files)) == 0
    assert len(train_files.intersection(val_files)) == 0
    assert len(test_files.intersection(val_files)) == 0

    # Check that all files are accounted for
    assert len(all_files) == len(train_files) + len(test_files) + len(val_files)
    assert len(all_files.intersection(train_files)) == len(train_files)
    assert len(all_files.intersection(test_files)) == len(test_files)
    assert len(all_files.intersection(val_files)) == len(val_files)

    # Check that the number of files is correct
    total_files = len(train_files) + len(test_files) + len(val_files)

    test_frac = len(test_files) / total_files
    val_frac = len(val_files) / total_files

    # Checks that the proportions are correct
    assert abs(test_frac - settings["data"]["split"]["test_frac"]) < 0.01
    assert abs(val_frac - settings["data"]["split"]["val_frac"]) < 0.01
