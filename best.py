import pickle
import json
import os
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.dir, "*.pkl"))
    metric_files = glob.glob(os.path.join(args.dir, "metrics", "*.pkl"))

    for ff, mf in zip(files, metric_files):
        with open(ff, "rb") as f:
            data = pickle.load(f)
        with open(mf, "rb") as f:
            metrics = pickle.load(f)

        best = metrics["selected_nb"]["balacc"]["val"].argmax()
        print(f"{ff}: {best}")

        setts = data[best]["changed_settings"]
        setts = {k: float(v) for k, v in setts.items()}

        print(json.dumps(setts, indent=4))
