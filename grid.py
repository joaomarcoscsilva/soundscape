import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from jax import numpy as jnp
import jax


def grid(x):
    x = np.pad(x, (0, 36 - len(x)), "constant")
    x = x.reshape(6, 6)
    return x


for k in sys.argv[1:]:
    with open(k, "rb") as f:
        x = pickle.load(f)

    sets = [i["changed_settings"] for i in x]

    mixup = np.array([i["mixup_alpha"] for i in sets])
    cutmix = np.array([i["cutmix_alpha"] for i in sets])
    balacc = np.array([i["metric"] for i in x])
    ce = np.array([i["logs"]["mean_val_ce_loss"][i["best_epoch"]] for i in x])
    labels = np.array([i["logs"]["val_labels"][i["best_epoch"]] for i in x])
    logits = np.array([i["logs"]["val_logits"][i["best_epoch"]] for i in x])

    probs = jax.nn.softmax(logits)
    entropy = -(probs * jnp.log(probs + 1e-5)).sum(axis=-1).mean(axis=-1)
    entropy = np.array(entropy)

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    sn.heatmap(grid(mixup), ax=ax[0][0], annot=True, fmt=".2f")
    sn.heatmap(grid(cutmix), ax=ax[0][1], annot=True, fmt=".2f")
    sn.heatmap(grid(balacc), ax=ax[1][0], annot=True, fmt=".2f")
    sn.heatmap(grid(ce), ax=ax[1][1], annot=True, fmt=".2f")
    sn.heatmap(grid(entropy), ax=ax[2][0], annot=True, fmt=".2f")
    ax[0][0].set_title("Mixup")
    ax[0][1].set_title("Cutmix")
    ax[1][0].set_title("Balacc")
    ax[1][1].set_title("CE")
    ax[2][0].set_title("Entropy")

    plt.savefig(k + "grid.png")
