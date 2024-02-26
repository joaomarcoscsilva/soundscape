import json
from copy import copy

import hydra
from jax import numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm


def _wrap_merge_fn(merge_fn):
    return lambda values: merge_fn(values) if isinstance(values, list) else values


class Logger:
    def __init__(
        self,
        pbar_len=0,
        pbar_keys=[],
        merge_fn="concat",
        pbar_level=0,
    ):
        self.pbar_keys = pbar_keys
        self.merge_fn = _wrap_merge_fn(
            jnp.concatenate if "concat" in merge_fn.lower() else jnp.stack
        )
        self.pbar_len = pbar_len
        self.pbar_level = pbar_level
        self._merged = None
        self.prefixes = set()

    def restart(self):
        self.pbar = tqdm(total=self.pbar_len, ncols=160, position=self.pbar_level)
        self._logs: dict[str, list] = {}
        self._merged = None
        self.prefixes = set()

    def _extract_keys(self, dictionaries, prefix):
        self.prefixes.add(prefix)

        for dictionary in dictionaries:
            for key in dictionary.keys():
                if prefix + key not in self._logs:
                    self._logs[prefix + key] = []

                val = dictionary[key]
                if jnp.ndim(val) == 0:
                    val = jnp.array([val])

                self._logs[prefix + key].append(val)

    def _update_pbar(self):
        descs = []

        merged = self.merge()
        for key in self.pbar_keys:
            if key in merged:
                mean = mean_keep_dtype(merged[key])
                descs.append(f"{key} {format_digits(mean)}")

        self.pbar.set_description(" █ ".join(descs))
        self.pbar.update()

    def update(self, *dictionaries, prefix=""):
        self._merged = None
        self._extract_keys(dictionaries, prefix)
        self._update_pbar()

    def merge(self):
        if self._merged is None:
            self._merged = {
                k: self.merge_fn(v) for k, v in self._logs.items() if len(v) > 0
            }
        return self._merged

    def close(self):
        self.pbar.close()
        return self.merge()

    def __getitem__(self, key):
        return self.merge()[key]


class TrainingLogger(Logger):
    def __init__(
        self,
        *args,
        nan_metrics=[],
        optimizing_metric=None,
        optimizing_mode="max",
        patience=None,
        saved_keys=[],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizing_metric = optimizing_metric
        self.optimizing_sign = 1 if optimizing_mode == "max" else -1
        self.patience = patience
        self.nan_metrics = set(nan_metrics)
        self.saved_keys = set(saved_keys)

    def _maximizing_metric(self, metrics=None):
        if self.optimizing_metric is None:
            return None

        metric = (
            self[self.optimizing_metric]
            if metrics is None
            else metrics[self.optimizing_metric]
        )
        metric = metric.mean(tuple(range(1, metric.ndim)))
        metric = self.optimizing_sign * metric
        return metric

    def best(self):
        if self.optimizing_metric is not None:
            return self._maximizing_metric().max()
        return None

    def latest(self):
        if self.optimizing_metric is not None:
            return self._maximizing_metric()[-1]
        return None

    def early_stop(self):
        for metric in self._list_keys(self.nan_metrics):
            if not jnp.isfinite(self[metric][-1]).all():
                return True

        if self.optimizing_metric is not None:
            metric = self._maximizing_metric()
            if jnp.argmax(metric) < len(metric) - self.patience:
                return True

        return False

    def improved(self):
        if self.optimizing_metric is not None:
            return self.latest() == self.best()

        return True

    def _list_keys(self, keys):
        return {
            prefix + k
            for k in keys
            for prefix in self.prefixes
            if prefix + k in self.merge()
        }

    def serialized(self):
        merged = self.merge()
        raw = {
            key: merged[key].round(6).tolist()
            for key in self._list_keys(self.saved_keys)
        }
        return json.dumps(raw, sort_keys=True)

    def best_epoch(self, metrics=None):
        metric = self._maximizing_metric(metrics)
        selected_epoch = -1 if metric is None else metric.argmax()
        return selected_epoch

    def best_epoch_metrics(self, sel_metrics=None, metrics=None):
        sel_metrics = self.merge() if sel_metrics is None else sel_metrics
        metrics = self.merge() if metrics is None else metrics

        best_epoch = self.best_epoch(sel_metrics)
        return {k: float(v[best_epoch].mean()) for k, v in metrics.items()}


def get_logger(settings, *args, **kwargs):
    return hydra.utils.instantiate(settings, *args, **kwargs)


def format_digits(val, digits=6):
    """
    Format a number to a given number of digits.
    """

    # Check if val is nan
    if jnp.isnan(val):
        return "   nan"

    # Check if val is inf or -inf
    if jnp.isinf(val):
        return "   inf" if val > 0 else "  -inf"

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
