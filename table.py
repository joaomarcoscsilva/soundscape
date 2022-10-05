import pickle
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--acc", action="store_true")

args = parser.parse_args()

files = [
    os.path.join(args.dir, "resnet_noaug_test.pkl"),
    os.path.join(args.dir, "resnet_aug_test.pkl"),
    os.path.join(args.dir, "vit_noaug_test.pkl"),
    os.path.join(args.dir, "vit_aug_test.pkl"),
]

data = []

for file in files:
    with open(file, "rb") as f:
        data.append(pickle.load(f))


models = ["ResNet", "ViT"]
augmentations = ["No", "Yes"]
calibrations = [
    "No",
    "Bias",
    "Temperature",
    "Temperature + Bias",
    "Vector",
    "Vector + Bias",
]
optimal_decisions = ["No", "Yes"]

model_map = {
    ("ResNet", "No"): data[0],
    ("ResNet", "Yes"): data[1],
    ("ViT", "No"): data[2],
    ("ViT", "Yes"): data[3],
}

calibration_map = {
    "No": "test",
    "Bias": "test_notemp_bias",
    "Temperature": "test_scalar_nobias",
    "Temperature + Bias": "test_scalar_bias",
    "Vector": "test_vector_nobias",
    "Vector + Bias": "test_vector_bias",
}


def acc_table():
    rows = 49
    cols = 6
    table = [["" for i in range(cols)] for j in range(rows)]

    table[0] = [
        "Model",
        "Augmentation",
        "Calibration",
        "Optimal Decision",
        "Balanced Accuracy",
        r"\\",
    ]

    for i in range(2):
        table[1 + i * 24][0] = models[i % 2]
        table[i * 24][5] = r"\\ \midrule"

    for i in range(4):
        table[1 + i * 12][1] = augmentations[i % 2]
        if table[i * 12][5] == "":
            table[i * 12][5] = r"\\ \cmidrule{2-5}"

    for i in range(24):
        table[1 + 2 * i][2] = calibrations[i % 6]
        if table[2 * i][5] == "":
            table[2 * i][5] = r"\\ \cmidrule{3-5}"

    for i in range(48):
        table[1 + i][3] = optimal_decisions[i % 2]
        if table[1 + i][5] == "":
            table[1 + i][5] = r"\\"

    model = augmentation = calibration = optimal_decision = None

    for row in table[1:]:
        model = row[0] if row[0] != "" else model
        augmentation = row[1] if row[1] != "" else augmentation
        calibration = row[2] if row[2] != "" else calibration
        optimal_decision = row[3] if row[3] != "" else optimal_decision

        model_data = model_map[(model, augmentation)]
        split = calibration_map[calibration]

        row[0] = f"\\multirow{{29}}{{*}}{{{model}}}" if row[0] != "" else ""
        row[1] = f"\\multirow{{12}}{{*}}{{{augmentation}}}" if row[1] != "" else ""
        row[2] = f"\\multirow{{2}}{{*}}{{{calibration}}}" if row[2] != "" else ""

        if optimal_decision == "No":
            row[4] = 100 * model_data["selected_nb"]["balacc_nb"][split]
        else:
            row[4] = 100 * model_data["selected_nb"]["balacc"][split]

        row[4] = f"${row[4].mean():.2f} \pm {row[4].std():.2f}$    "

    collens = [max(len(row[i]) for row in table) for i in range(cols)]

    for row in table:
        for i in range(cols):
            row[i] = row[i].ljust(collens[i])

    print(r"\begin{table}[ht]")
    print(r"\caption{CAPTION}")
    print(r"\label{tab:LABEL}")
    print(r"\centering")
    print(r"\begin{tabular}{@{}ccccc@{}}")
    print(r"\toprule")

    for row in table:
        print(*row[:-1], sep=" & ", end=" ")
        print(row[-1])

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def metrics_table(metric_keys, metric_names):
    rows = 25
    cols = 4 + len(metric_keys)
    table = [["" for i in range(cols)] for j in range(rows)]

    table[0] = [
        "Model",
        "Augmentation",
        "Calibration",
        *metric_names,
        r"\\",
    ]

    for i in range(2):
        table[1 + i * 12][0] = models[i % 2]
        table[i * 12][-1] = r"\\ \midrule"

    for i in range(4):
        table[1 + i * 6][1] = augmentations[i % 2]
        if table[i * 6][-1] == "":
            table[i * 6][-1] = f"\\\\ \\cmidrule{{2-{cols-1}}}"

    for i in range(24):
        table[1 + i][2] = calibrations[i % 6]
        if table[1 + i][-1] == "":
            table[1 + i][-1] = r"\\"

    model = augmentation = calibration = None

    for row in table[1:]:
        model = row[0] if row[0] != "" else model
        augmentation = row[1] if row[1] != "" else augmentation
        calibration = row[2] if row[2] != "" else calibration

        model_data = model_map[(model, augmentation)]
        split = calibration_map[calibration]

        row[0] = f"\\multirow{{12}}{{*}}{{{model}}}" if row[0] != "" else ""
        row[1] = f"\\multirow{{6}}{{*}}{{{augmentation}}}" if row[1] != "" else ""

        for i, key in enumerate(metric_keys):
            row[3 + i] = model_data["selected_nb"][key][split]
            row[3 + i] = f"${row[3+i].mean():.4f} \pm {row[3+i].std():.4f}$    "

    collens = [max(len(row[i]) for row in table) for i in range(cols)]

    for row in table:
        for i in range(cols):
            row[i] = row[i].ljust(collens[i])

    print(r"\begin{table}[ht]")
    print(r"\caption{CAPTION}")
    print(r"\label{tab:LABEL}")
    print(r"\centering")
    print(f"\\begin{{tabular}}{{@{{}}{'c' * (cols-1)}@{{}}}}")
    print(r"\toprule")

    for row in table:
        print(*row[:-1], sep=" & ", end=" ")
        print(row[-1])

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if args.acc:
    acc_table()
else:
    metrics_table(
        ["ce", "brier", "ece", "entropy"],
        ["Cross-Entropy", "Brier Score", "ECE", "Entropy"],
    )
