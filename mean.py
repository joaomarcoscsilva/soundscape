import pickle
import jax
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

from soundscape import dataset, settings

with settings.Settings.from_file("settings/supervised.yaml"):
    ws = dataset.get_class_weights()

np.set_printoptions(formatter={"float": lambda x: "{0:0.8f}".format(x)})

for k in sys.argv[1:]:
    print(k)

    with open(k, "rb") as f:
        x = pickle.load(f)

    balacc = np.array([i["metric"] for i in x])
    ce = np.array([i["logs"]["mean_val_ce_loss"][i["best_epoch"]] for i in x])
    labels = np.array([i["logs"]["val_labels"][i["best_epoch"]] for i in x])
    logits = np.array([i["logs"]["val_logits"][i["best_epoch"]] for i in x])
    ids = np.array([i["logs"]["val_id"][i["best_epoch"]] for i in x])
    preds = np.array([i["logs"]["val_preds"][i["best_epoch"]] for i in x])

    probs = jax.nn.softmax(logits)
    entropy = -(probs * np.log(probs + 1e-5)).sum(axis=-1).mean(axis=-1)
    entropy = np.array(entropy)

    print()

    print("balacc\t", (balacc), f"({balacc.mean():0.8f}) +/- {balacc.std():0.8f})")
    print("celoss\t", (ce), f"({ce.mean():0.8f}) +/- {ce.std():0.8f})")
    print("entrpy\t", (entropy), f"({entropy.mean():0.8f}) +/- {entropy.std():0.8f})")

    # for k in range(1, 12):
    #     topk = []
    #     for i in range(5):
    #         tk = top_k_accuracy_score(
    #             labels[i], probs[i] * ws, k=k, sample_weight=ws[labels[i]]
    #         )
    #         topk.append(tk)
    #     topk = np.array(topk)
    #     print(f"top{k}\t", topk, f"({topk.mean():0.8f}) +/- {topk.std():0.8f})")

    print()
