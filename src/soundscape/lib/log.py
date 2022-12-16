from jax import numpy as jnp
import copy
import pickle
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from . import utils, constants


at_least_one_dim = lambda l: [x if x.ndim > 0 else x[None] for x in l]


def running_average_desc_fn(prefix):
    """
    Create a description function for the training loop.
    """

    def desc(log_state, aux):
        log_state = {
            k: v for k, v in log_state.items() if k not in ["predictions", "labels", "probabilities", "indices"]
        }
        log_state = utils.dict_map(at_least_one_dim, log_state)
        log_state = utils.dict_map(jnp.concatenate, log_state)
        log_state = utils.dict_map(jnp.mean, log_state)

        s = prefix
        for k in sorted(log_state.keys()):
            s += f" {k}: {log_state[k]:.3f} "

        return s

    return desc


class Logger:
    """
    Logger object that saves training data to tsv files.
    """

    def __init__(self, optimizing_metric=None):
        self.data = {}
        self._step = 0
        self.last_saved = {}
        self._files = {}
        self.optimizing_metric = optimizing_metric

    def log(self, key, value, once=False):
        """
        Log a key-value pair.
        If once is False, the value is appended to a list of values.
        """

        if key not in self.data:
            self.data[key] = []
            self.last_saved[key] = 0

        if not once:
            self.data[key].append((self._step, value))
        else:
            self.data[key] = [(self._step, value)]

    def is_best(self):
        """
        Check if the current value of the optimizing metric is the best so far.
        If the optimizing metric is not set, return True.
        """

        if self.optimizing_metric is None:
            return True

        sign = 1 if self.optimizing_metric[0] == "+" else "-"

        vals = [sign * v for _, v in self.data[self.optimizing_metric[1:]]]

        return max(vals) == vals[-1]

    def consolidate(
        self, to_pickle, confusion_labels_key=None, confusion_preds_key=None
    ):
        """
        Update the metrics.json file with the current values and saves
        a checkpoint of the model in models/model.pkl.
        """

        metrics = {"step": self._step}

        for key in self.data:
            if "eval" in key:
                v = self.data[key][-1][1]
                if isinstance(v, jnp.ndarray) and v.ndim == 0:
                    metrics[key] = float(v)

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        with open(f"models/model.pkl", "wb") as f:
            pickle.dump(to_pickle, f)

        if confusion_labels_key is not None and confusion_preds_key is not None:
            self.confusion_matrix(
                "eval_labels", "eval_predictions", file="plots/best_confusion.png"
            )

    def step(self):
        """
        Increment the step counter.
        """

        self._step += 1

    def save(self, to_pickle=None, confusion_labels_key=None, confusion_preds_key=None):
        """
        Save the logged data to tsv files.
        Returns True if the model was saved.
        """

        for key, v in self.data.items():

            if key not in self._files:
                self._files[key] = open(f"plots/{key}.tsv", "w")
                self._files[key].write(f"step\t{key}\n")

            f = self._files[key]

            for step, value in v[self.last_saved[key] :]:
                f.write(f"{step}\t{value}\n")
                f.flush()

            self.last_saved[key] = len(v)

        if self.is_best():
            self.consolidate(to_pickle, confusion_labels_key, confusion_preds_key)

    def confusion_matrix(self, labels_key, preds_key, file=None):
        """
        Plot a confusion matrix using the latest saved versions of labels_key and preds_key.
        """

        labels = self.data[labels_key][-1][1]
        preds = self.data[preds_key][-1][1]

        conf_mat = confusion_matrix(labels, preds, normalize="true")

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt=".2f",
            vmin=0.0,
            vmax=1.0,
            xticklabels=constants.CLASS_NAMES,
            yticklabels=constants.CLASS_NAMES,
        )

        plt.xlabel("Predicted")
        plt.ylabel("True")

        if file is None:
            file = f"plots/confusion/{self._step}.png"

        plt.savefig(file)
        plt.close()

    def __del__(self):
        """
        Close the files when the logger is deleted.
        """

        for f in self._files.values():
            f.close()


def log_dict(logger, d, prefix=""):
    """
    Log a dictionary.
    """

    for k, v in d.items():
        logger.log(prefix + k, v)


def batch_log_fn(logger, prefix=""):
    """
    Log results at the end of a training step.
    """

    def _log(log_state):
        if logger is None:
            return

        last_logs = utils.dict_map(lambda x: x[-1], log_state)
        last_logs = utils.dict_map(jnp.mean, last_logs)
        log_dict(logger, last_logs, prefix=prefix)

        logger.step()

    return _log


def epoch_log_fn(logger, prefix=""):
    """
    Log results at the end of a validation epoch.
    """

    def _log(log_state, to_pickle=None):
        if logger is None:
            return

        log_state = utils.dict_map(at_least_one_dim, log_state)
        log_state = utils.dict_map(jnp.concatenate, log_state)

        if "predictions" in log_state.keys():
            log_state = copy.deepcopy(log_state)

            preds = log_state.pop("predictions")
            labels = log_state.pop("labels")
            probs = log_state.pop("probabilities")
            indices = log_state.pop("indices")

            logger.log(prefix + "predictions", preds.tolist())
            logger.log(prefix + "labels", labels.tolist(), once=True)
            logger.log(prefix + "probabilities", probs.tolist())
            logger.log(prefix + "indices", indices.tolist(), once=True)

            logger.confusion_matrix(prefix + "labels", prefix + "predictions")

        log_state = utils.dict_map(jnp.mean, log_state)

        log_dict(logger, log_state, prefix)

        logger.save(to_pickle, prefix + "labels", prefix + "predictions")

    return _log
