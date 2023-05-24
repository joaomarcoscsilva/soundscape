import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from tqdm import tqdm
import os
from soundscape import settings, dataset
from sklearn.metrics import confusion_matrix
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with settings.Settings.from_file("settings/supervised.yaml"):
    ws = dataset.get_class_weights()
num_classes = 12

sn.set_theme()
sn.set_context("paper")
sn.set_style("darkgrid")

plt.rcParams["figure.dpi"] = 300

# plt.style.use("seaborn")
tex_fonts = {
    "text.usetex": True,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"],
}
plt.rcParams.update(tex_fonts)


def lineplot(y, label, ax=None):
    x = np.arange(y.shape[1])
    x = np.repeat(x, y.shape[0])
    y = y.T.reshape(-1)
    sn.lineplot(x=x, y=y, label=label, ax=ax)


def set_size(fraction, subplots=(1, 1)):
    textwidth = 430.00462  # pt
    fig_width = textwidth * fraction
    inches_per_pt = 1.0 / 72.27
    golden_ratio = (np.sqrt(5) - 1.0) / 2.0
    fig_width = fig_width * inches_per_pt
    fig_height = fig_width * golden_ratio * (subplots[0] / subplots[1])
    fig_dims = (fig_width, fig_height)
    return fig_dims


def plot5(x, plots_dir):
    plt.figure(figsize=set_size(1.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size(1.0, (1, 2)), sharey=True)

    ax1.set_title("Without Augmentation")
    ax1.set_ylabel("Cross Entropy")
    ax1.set_xlabel("Epoch")
    lineplot(x["vit_noaug_test"]["ce"]["train"], "Train", ax1)
    lineplot(x["vit_noaug_test"]["ce"]["val"], "Validation", ax1)

    ax2.set_title("With Augmentation")
    ax2.set_xlabel("Epoch")
    ax2.tick_params(axis="y", labelleft=True)
    lineplot(x["vit_aug_test"]["ce"]["train"], "Train", ax2)
    lineplot(x["vit_aug_test"]["ce"]["val"], "Validation", ax2)

    plt.legend()
    plt.savefig(f"{plots_dir}/ce_aug.svg")
    plt.clf()

    plt.figure(figsize=set_size(1.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size(1.0, (1, 2)), sharey=True)

    ax1.set_title("Without Augmentation")
    ax1.set_ylabel("Predictive Entropy")
    ax1.set_xlabel("Epoch")
    lineplot(x["vit_noaug_test"]["entropy"]["train"], "Train", ax1)
    lineplot(x["vit_noaug_test"]["entropy"]["val"], "Validation", ax1)

    ax2.set_title("With Augmentation")
    ax2.set_xlabel("Epoch")
    ax2.tick_params(axis="y", labelleft=True)
    lineplot(x["vit_aug_test"]["entropy"]["train"], "Train", ax2)
    lineplot(x["vit_aug_test"]["entropy"]["val"], "Validation", ax2)

    plt.legend()
    plt.savefig(f"{plots_dir}/entropy_aug.svg")
    plt.clf()

    plt.figure(figsize=set_size(1.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size(1.0, (1, 2)), sharey=True)

    ax1.set_title("Without Augmentation")
    ax1.set_xlabel("Epoch")
    lineplot(
        x["vit_noaug_test"]["entropy"]["val"], "Validation Predictive Entropy", ax1
    )
    lineplot(x["vit_noaug_test"]["ce"]["val"], "Validation Cross Entropy", ax1)
    lineplot(
        x["vit_noaug_test"]["ce"]["val_vector_bias"],
        "Validation Calibrated Entropy",
        ax1,
    )

    ax2.set_title("With Augmentation")
    ax2.set_xlabel("Epoch")
    ax2.tick_params(axis="y", labelleft=True)
    lineplot(x["vit_aug_test"]["entropy"]["val"], "Validation Predictive Entropy", ax2)
    lineplot(x["vit_aug_test"]["ce"]["val"], "Validation Cross Entropy", ax2)
    lineplot(
        x["vit_aug_test"]["ce"]["val_vector_bias"], "Validation Calibrated Entropy", ax2
    )

    plt.legend()
    plt.savefig(f"{plots_dir}/entropy_ce_aug.svg")
    plt.clf()

    plt.figure(figsize=set_size(1.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size(1.0, (1, 2)), sharey=True)

    ax1.set_title("Without Augmentation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Temperature")
    lineplot(
        x["vit_noaug_test"]["calibration"]["scalar_bias"]["w"][..., 0], "Train", ax1
    )

    ax2.set_title("With Augmentation")
    ax2.set_xlabel("Epoch")
    ax2.tick_params(axis="y", labelleft=True)
    lineplot(x["vit_aug_test"]["calibration"]["scalar_bias"]["w"][..., 0], "Train", ax2)

    plt.legend()
    plt.savefig(f"{plots_dir}/temperature_aug.svg")
    plt.clf()


def plot180(x, plots_dir):
    from IPython import embed

    embed()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("filenames", nargs="+")
    parser.add_argument("--dataset", default="leec")

    args = parser.parse_args()

    x = {
        os.path.basename(filename).replace(".pkl", ""): pickle.load(
            open(filename, "rb")
        )
        for filename in args.filenames
    }

    plots_dir = os.path.join(os.path.dirname(args.filenames[0]), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot5(x, plots_dir)

    if len(x) == 180:
        plot180(x, plots_dir)
