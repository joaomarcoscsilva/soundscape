import pickle

from jax import numpy as jnp
from tqdm import tqdm


class Logger:
    def __init__(
        self,
        keys,
        pbar_len=0,
        pbar_keys=[],
        merge_fn="concat",
        pbar_position=0,
    ):
        self.keys = set(keys)
        self.pbar_keys = pbar_keys
        self.merge_fn = jnp.concatenate if "concat" in merge_fn.lower() else jnp.stack
        self.pbar_len = pbar_len
        self.pbar_position = pbar_position

    def restart(self):
        self.pbar = tqdm(total=self.pbar_len, ncols=160, position=self.pbar_position)
        self.logs: dict[str, list] = {k: [] for k in self.keys}

    def _extract_keys(self, dictionaries, prefix):
        for dictionary in dictionaries:
            for key in self.keys & dictionary.keys():
                self.logs[prefix + key].append(dictionary[key])

    def _update_pbar(self, prefix):
        descs = []

        for key in self.pbar_keys:
            mean = mean_keep_dtype(jnp.concatenate(self.logs[key]))
            descs.append(f"{prefix+key} {format_digits(mean)}")

        self.pbar.set_description(" █ ".join(descs))
        self.pbar.update()

    def update(self, *dictionaries, prefix=""):
        self._extract_keys(dictionaries, prefix)
        self._update_pbar(prefix)

    def merge(self):
        return {k: self.merge_fn(v) for k, v in self.logs.items() if len(v) > 0}

    def close(self):
        self.pbar.close()
        return self.merge()

    def pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.merge(), f)


def format_digits(val, digits=6):
    """
    Format a number to a given number of digits.
    """

    # Check if val is nan
    if jnp.isnan(val):
        return "   nan"

    # Get the number of digits in the integer part
    integer_len = len(str(int(val)))

    if val.dtype in [jnp.int32, jnp.int64]:
        # If an integer, pad with spaces
        return f"{val:{digits}d}"
    else:
        # If a float, assign all remaining digits to the fractional part
        return f"{val:.{max(digits - integer_len - 1, 0)}f}"


def mean_keep_dtype(x):
    """
    Compute the mean of an array, without changing its dtype.
    """
    return x.mean(dtype=x.dtype)


# def save_params(
#     values, *, name, save_weights, early_stopping, optimizing_metric, optimizing_mode
# ):
#     """
#     Save the parameters of the model to a file.
#     """

#     if not save_weights:
#         return values

#     keys = ["params", "fixed_params", "state"]
#     to_save = {k: values[k] for k in keys if k in values}

#     save = not early_stopping

#     if early_stopping:
#         metric = values["_epoch_logs"][optimizing_metric]

#         if optimizing_mode == "min":
#             metric = -metric

#         if metric.ndim != 1:
#             raise ValueError(
#                 f"Metric {optimizing_metric} should be be a mean over the epoch."
#             )

#         if metric[-1] == metric.max():
#             save = True

#     if save:
#         with open(f"params/{name}.pkl", "wb") as f:
#             pickle.dump(to_save, f)

#     return values


# def stop_if_nan(nan_metrics):
#     """
#     Return a composable function that stops the training if any of the given
#     metrics is NaN.
#     """

#     def _stop_if_nan(values):
#         for metric in nan_metrics:
#             if jnp.isnan(values["_epoch_logs"][metric][-1]).any():
#                 return {**values, "_stop": f"NaN in {metric}"}

#         return values

#     return _stop_if_nan


# def stop_if_no_improvement(
#     *, optimizing_mode, optimizing_metric, give_up_threshold, give_up_after
# ):
#     """
#     Return a composable function that stops the training if the optimizing metric
#     is below a threshold after the given number of epochs.
#     """

#     sign = 1 if optimizing_mode == "max" else -1

#     def _stop_if_no_improvement(values):
#         if values["epoch"] % give_up_after == 0:
#             last_metric = values["_epoch_logs"][optimizing_metric][-1]
#             if sign * last_metric < sign * give_up_threshold:
#                 return {**values, "_stop": f"Slow convergence"}

#         return values

#     return _stop_if_no_improvement
