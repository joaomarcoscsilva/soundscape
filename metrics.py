import pickle
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import jax
import os
import optax
from soundscape import settings, dataset
import tensorflow_probability as tfp
from jax import numpy as jnp
from sklearn.metrics import confusion_matrix
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sn.set_theme()
sn.set_context("paper")
sn.set_style("darkgrid")

plt.rcParams["figure.dpi"] = 300

with settings.Settings.from_file("settings/supervised.yaml"):
    ws = dataset.get_class_weights()

def transform(params, logits):
    return logits * params["w"] + params["b"]


def ce(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    return (optax.softmax_cross_entropy(logits, one_hot_labels)).mean()


def brier(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    probs = jax.nn.softmax(logits)
    return ((probs - one_hot_labels) ** 2).mean()


def balacc(logits, labels):
    probs = jax.nn.softmax(logits) * ws
    return ((labels == probs.argmax(-1)) * ws[labels]).mean()


def balacc_nb(logits, labels):
    probs = jax.nn.softmax(logits)
    return ((labels == probs.argmax(-1)) * ws[labels]).mean()


def acc(logits, labels):
    probs = jax.nn.softmax(logits) * ws
    return (labels == probs.argmax(-1)).mean()


def acc_nb(logits, labels):
    probs = jax.nn.softmax(logits)
    return (labels == probs.argmax(-1)).mean()


def entropy(logits, labels=None):
    probs = jax.nn.softmax(logits)
    probs = probs + 1e-5
    return -(probs * jnp.log(probs)).sum(axis=-1).mean()


def ECE(logits, labels, n_bins=5):
    return tfp.substrates.jax.stats.expected_calibration_error(n_bins, logits, labels)


def confusion_nb(logits, labels):
    probs = jax.nn.softmax(logits)
    preds = probs.argmax(-1)
    return np.stack(
        [confusion_matrix(l, p, normalize="true") for l, p in zip(labels, preds)]
    )


def confusion(logits, labels):
    probs = jax.nn.softmax(logits) * ws
    preds = probs.argmax(-1)
    return np.stack(
        [confusion_matrix(l, p, normalize="true") for l, p in zip(labels, preds)]
    )


transform_batch = jax.vmap(jax.vmap(transform))
entropy_batch = jax.vmap(jax.vmap(entropy))


def filterclass(logits, labels, cls):
    ids = labels == cls
    return logits[ids], labels[ids]


def stack_array(x, *ks):
    def gt(v):
        for k in ks:
            v = v[k]
        v = np.pad(v, ((0, 512 - v.shape[0]), *[(0, 0)] * (v.ndim - 1)), mode="edge")
        return v

    return np.stack([gt(i) for i in x])


@jax.vmap
@jax.vmap
def reorder_id(x, ids):
    return x[ids.argsort()]


def load_file(filename):
    with open(filename, "rb") as f:
        x = pickle.load(f)

    params = {
        cal_type: {
            "w": stack_array(x, "calibration", cal_type, "w"),
            "b": stack_array(x, "calibration", cal_type, "b"),
        }
        for cal_type in x[0]["calibration"]
    }

    train_logits = stack_array(x, "logs", "logits")
    # train_labels = stack_array(x, "logs", "labels")
    train_one_hot_labels = stack_array(x, "logs", "one_hot_labels")
    # train_id = stack_array(x, "logs", "id")

    val_logits = stack_array(x, "logs", "val_logits")
    # val_labels = stack_array(x, "logs", "val_labels")
    val_one_hot_labels = stack_array(x, "logs", "val_one_hot_labels")
    # val_id = stack_array(x, "logs", "val_id")

    if "test_logits" in x[0]["logs"]:
        test_logits = stack_array(x, "logs", "test_logits")
        # test_labels = stack_array(x, "logs", "test_labels")
        test_one_hot_labels = stack_array(x, "logs", "test_one_hot_labels")
        # test_id = stack_array(x, "logs", "test_id")
    else:
        test_logits = val_logits
        # test_labels = val_labels
        test_one_hot_labels = val_one_hot_labels
        # test_id = val_id

    train_labels = train_one_hot_labels.argmax(-1)
    val_labels = val_one_hot_labels.argmax(-1)
    test_labels = test_one_hot_labels.argmax(-1)

    # train_logits = reorder_id(train_logits, train_id)
    # train_labels = reorder_id(train_labels, train_id)
    # train_one_hot_labels = reorder_id(train_one_hot_labels, train_id)

    # val_logits = reorder_id(val_logits, val_id)
    # val_labels = reorder_id(val_labels, val_id)
    # val_one_hot_labels = reorder_id(val_one_hot_labels, val_id)

    # test_logits = reorder_id(test_logits, test_id)
    # test_labels = reorder_id(test_labels, test_id) 
    # test_one_hot_labels = reorder_id(test_one_hot_labels, test_id)

    train_cal_logits = {
        cal_type: transform_batch(params[cal_type], train_logits) for cal_type in params
    }

    val_cal_logits = {
        cal_type: transform_batch(params[cal_type], val_logits) for cal_type in params
    }

    test_cal_logits = {
        cal_type: transform_batch(params[cal_type], test_logits) for cal_type in params
    }

    return {
        "logits": {
            "train": train_logits,
            "val": val_logits,
            "test": test_logits,
            **{f"train_{cal_type}": train_cal_logits[cal_type] for cal_type in params},
            **{f"val_{cal_type}": val_cal_logits[cal_type] for cal_type in params},
            **{f"test_{cal_type}": test_cal_logits[cal_type] for cal_type in params},
        },
        "labels": {
            "train": train_labels,
            "val": val_labels,
            "test": test_labels,
            **{f"train_{cal_type}": train_labels for cal_type in params},
            **{f"val_{cal_type}": val_labels for cal_type in params},
            **{f"test_{cal_type}": test_labels for cal_type in params},
        },
        "one_hot_labels": {
            "train": train_one_hot_labels,
            "val": val_one_hot_labels,
            "test": test_one_hot_labels,
            **{f"train_{cal_type}": train_one_hot_labels for cal_type in params},
            **{f"val_{cal_type}": val_one_hot_labels for cal_type in params},
            **{f"test_{cal_type}": test_one_hot_labels for cal_type in params},
        },
        "calibration": params,
    }


def compute_metrics(x, noplot=False, select_balacc=True):
    n = len(x["logits"]["train"])

    splits = x["logits"].keys()
    metrics = ["ce", "brier", "acc", "acc_nb", "balacc", "balacc_nb", "entropy", "ece"]
    fns = [ce, brier, acc, acc_nb, balacc, balacc_nb, entropy, ECE]
    batch_fns = [jax.vmap(jax.vmap(fn)) for fn in fns]

    x["perclass"] = {}

    x["selected"] = {}
    x["selected_nb"] = {}
    x["selected_last"] = {}

    x["confusion"] = {}
    x["confusion_nb"] = {}

    for metric, fn in zip(metrics, batch_fns):
        x[metric] = {}

        for split in splits:
            x[metric][split] = fn(x["logits"][split], x["labels"][split])

    x["entropy"]["train_labels"] = entropy_batch(jnp.log(x["one_hot_labels"]["train"]))

    x["selected_epochs"] = {
        split: x["balacc" if select_balacc else "acc"][
            split.replace("train", "val").replace("test", "val")
        ].argmax(-1)
        for split in splits
    }

    x["selected_epochs_nb"] = {
        split: x["balacc_nb" if select_balacc else "acc_nb"][
            split.replace("train", "val").replace("test", "val")
        ].argmax(-1)
        for split in splits
    }

    def sel(v, e):
        return v[np.arange(n), e]

    for metric in metrics + ["logits", "labels", "one_hot_labels"]:
        x["selected"][metric] = {
            split: sel(x[metric][split], x["selected_epochs"][split])
            for split in splits
        }

        x["selected_nb"][metric] = {
            split: sel(x[metric][split], x["selected_epochs_nb"][split])
            for split in splits
        }

        x["selected_last"][metric] = {
            split: sel(x[metric][split], -1) for split in splits
        }

    x["selected"]["entropy"]["train_labels"] = {
        split: sel(x["entropy"]["train_labels"], x["selected_epochs"][split])
        for split in splits
    }

    x["selected"]["calibration"] = {
        split: sel(
            x["calibration"][split.split("_", 1)[1]]["w"][..., 0],
            x["selected_epochs"][split],
        )
        for split in splits
        if "_" in split
    }

    if not noplot:
        for k in x["logits"]:
            epoch_labels = x["selected"]["labels"][k]
            epoch_logits = x["selected"]["logits"][k]
            x["confusion_nb"][k] = confusion_nb(epoch_logits, epoch_labels)

            epoch_labels = x["selected"]["labels"][k]
            epoch_logits = x["selected"]["logits"][k]
            x["confusion"][k] = confusion(epoch_logits, epoch_labels)

        for metric, fn in zip(metrics, fns):
            x["perclass"][metric] = {}

            for k in x["logits"]:
                x["perclass"][metric][k] = []

                logits = x["selected"]["logits"][k]
                labels = x["selected"]["labels"][k]

                for cls in range(logits.shape[-1]):
                    x["perclass"][metric][k].append([])

                    for i in range(n):
                        class_logits, class_labels = filterclass(
                            logits[i], labels[i], cls
                        )
                        x["perclass"][metric][k][cls].append(
                            fn(class_logits, class_labels)
                        )

                x["perclass"][metric][k] = np.array(x["perclass"][metric][k])

    return x


def lineplot(y, label):
    x = np.arange(y.shape[1])
    x = np.repeat(x, y.shape[0])
    y = y.T.reshape(-1)
    sn.lineplot(x=x, y=y, label=label)


def print_results(x, onlytest=False):
    n = len(x["logits"]["train"])

    def pad_str(s, l):
        return " " * (l - len(s)) + s

    def print_metric(split, metrics):
        for metric in metrics:
            mean = x["selected"][metric][split].mean()
            std = x["selected"][metric][split].std()

            if "acc" in metric:
                mean *= 100
                std *= 100
                prec = ".2f"
            elif "brier" in metric:
                prec = ".4f"
            elif "entropy" in metric:
                prec = ".3f"
            else:
                prec = ".4f"

            print(
                f"{pad_str(split, 15)} {pad_str(metric, 15)}: {mean:{prec}} +- {std:{prec}}"
            )
        print()

    metrics = ["acc_nb", "acc", "entropy", "ece", "ce", "brier", "balacc_nb", "balacc"]

    for split in x["acc"].keys():
        if onlytest and "test" not in split:
            continue

        print_metric(split, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("filenames", nargs="+")
    parser.add_argument("--noplot", action="store_true", default=False)
    parser.add_argument('--nosave', action='store_true', default=False)
    parser.add_argument("--dataset", default="leec")
    parser.add_argument("--onlytest", action="store_true", default=False)

    args = parser.parse_args()

    if args.dataset == "leec":
        with settings.Settings.from_file("settings/supervised.yaml"):
            ws = dataset.get_class_weights()
        num_classes = 12
    elif args.dataset == "leec2":
        with settings.Settings.from_file("settings/supervised.yaml"):
            ws = dataset.get_class_weights(num_classes=2)
        num_classes = 2
    else:
        with settings.Settings.from_file(f"settings/supervised.yaml"):
            with settings.Settings.from_file(f"settings/{args.dataset}.yaml"):
                ws = dataset.get_class_weights()
        num_classes = len(ws)

    for filename in args.filenames:
        x = load_file(filename)
        x = compute_metrics(x, select_balacc=args.dataset == "leec", noplot=args.noplot)

        print_results(x, onlytest=args.onlytest)

        new_filename = os.path.join(
            os.path.dirname(filename), "metrics", os.path.basename(filename)
        )

        if not args.nosave:
            os.makedirs(os.path.dirname(new_filename), exist_ok=True)
            with open(new_filename, "wb") as f:
                pickle.dump(x, f)

        if args.noplot:
            continue
