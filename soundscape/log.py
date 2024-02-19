import json

import hydra
from jax import numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm


class Logger:
    def __init__(
        self,
        keys,
        pbar_len=0,
        pbar_keys=[],
        merge_fn="concat",
        pbar_level=0,
        saved_keys=set(),
    ):
        self.keys = set(keys)
        self.pbar_keys = pbar_keys
        self.merge_fn = jnp.concatenate if "concat" in merge_fn.lower() else jnp.stack
        self.pbar_len = pbar_len
        self.pbar_level = pbar_level
        self.saved_keys = set(saved_keys)
        self._merged = None

    def restart(self):
        self.pbar = tqdm(total=self.pbar_len, ncols=160, position=self.pbar_level)
        self.logs: dict[str, list] = {k: [] for k in self.keys}
        self._merged = None

    def _extract_keys(self, dictionaries, prefix):
        for dictionary in dictionaries:
            for key in self.keys & dictionary.keys():
                if prefix + key not in self.logs:
                    self.logs[prefix + key] = []

                val = dictionary[key]
                if jnp.ndim(val) == 0:
                    val = jnp.array([val])

                self.logs[prefix + key].append(val)

    def _update_pbar(self):
        descs = []

        merged = self.merge()
        for key in merged.keys() & self.pbar_keys:
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
                k: self.merge_fn(v) for k, v in self.logs.items() if len(v) > 0
            }
        return self._merged

    def close(self):
        self.pbar.close()
        return self.merge()

    def serialized(self):
        merged = self.merge()
        raw = {k: merged[k].round(6).tolist() for k in self.saved_keys & merged.keys()}
        return json.dumps(raw, sort_keys=True)

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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nan_metrics = nan_metrics
        self.optimizing_metric = optimizing_metric
        self.optimizing_sign = 1 if optimizing_mode == "max" else -1
        self.patience = patience

    def early_stop(self):
        for metric in self.nan_metrics:
            if jnp.isnan(self[metric][-1]).any():
                return True

        if self.optimizing_metric is not None:
            metric = self[self.optimizing_metric]
            metric = metric.mean(range(1, metric.ndim))
            metric = self.optimizing_sign * metric

            if jnp.argmax(metric) < len(metric) - self.patience:
                return True

        return False


def get_logger(settings, *args, **kwargs):
    return hydra.utils.instantiate(settings, *args, **kwargs)


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
