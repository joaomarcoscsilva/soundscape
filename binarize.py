import argparse
import os
import pickle
import jax
from jax import numpy as jnp


def binarize_logits(logits):
    probs = jax.nn.softmax(logits)
    class0 = probs[..., :6].sum(axis=-1)
    class1 = probs[..., 6:].sum(axis=-1)
    probs = jnp.stack([class0, class1], axis=-1)
    return jnp.log(probs)


def bin_labels(labels):
    return jnp.int32(labels > 5)


def bin_one_hot(one_hot_labels):
    class0 = one_hot_labels[:, :6].sum(axis=1)
    class1 = one_hot_labels[:, 6:].sum(axis=1)
    one_hot_labels = jnp.stack([class0, class1], axis=1)
    return one_hot_labels


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("files", nargs="+")
    argparser.add_argument("--gpu", action="store_true", default=False)
    argparser.add_argument("--target", default="bin")
    args = argparser.parse_args()

    for k in args.files:
        with open(k, "rb") as f:
            x = pickle.load(f)

        for i in range(len(x)):
            x[i]["logs"]["logits"] = binarize_logits(x[i]["logs"]["logits"])
            x[i]["logs"]["val_logits"] = binarize_logits(x[i]["logs"]["val_logits"])
            x[i]["logs"]["test_logits"] = binarize_logits(x[i]["logs"]["test_logits"])

            x[i]["logs"]["labels"] = bin_labels(x[i]["logs"]["labels"])
            x[i]["logs"]["val_labels"] = bin_labels(x[i]["logs"]["val_labels"])
            x[i]["logs"]["test_labels"] = bin_labels(x[i]["logs"]["test_labels"])

            x[i]["logs"]["one_hot_labels"] = bin_one_hot(x[i]["logs"]["one_hot_labels"])
            x[i]["logs"]["val_one_hot_labels"] = bin_one_hot(
                x[i]["logs"]["val_one_hot_labels"]
            )
            x[i]["logs"]["test_one_hot_labels"] = bin_one_hot(
                x[i]["logs"]["test_one_hot_labels"]
            )

        dirname = os.path.dirname(k)
        basename = os.path.basename(k)
        os.makedirs(f"{dirname}/{args.target}", exist_ok=True)

        with open(f"{dirname}/{args.target}/{basename}", "wb") as f:
            pickle.dump(x, f)
