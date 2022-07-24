import pandas as pd
from tqdm import tqdm
import sys, os

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python check_labels.py <label-file> <wav-dir>"

    labels_file = sys.argv[1]
    wav_dir = sys.argv[2]

    df = pd.read_csv(labels_file)

    subdirs = os.listdir(wav_dir)

    files = set()
    for d in subdirs:
        files.update(os.listdir(os.path.join(wav_dir, d)))

    labelled_files = set(df["file"].values)

    missing = labelled_files - files

    print("\n", len(missing), "/", len(labelled_files), "files missing")
