from .composition import Composable
from . import settings

import numpy as np
from tqdm import tqdm
from jax import numpy as jnp
import jax
from tqdm import tqdm
from itertools import chain
import pickle
import haiku as hk


def track(keys, prefix=""):
    """
    Return a composable function that tracks the values of the given keys after each step.
    The values are stored in a list of dictionaries, one for each step, in the "_logs" key.

    Parameters:
    ----------
    keys: list of str
        The keys to track.
    prefix: str
        A prefix to add to the keys before storing them in the logs.
    """

    @Composable
    def _track(values):
        # Get the values to be logged
        step_logs = {prefix + key: values[key] for key in keys}

        # Get the logs list, or create a new one
        logs = values.get("_logs", [])

        # Append the new logs to the list
        logs.append(step_logs)

        return {**values, "_logs": logs}

    return _track


@Composable
def count_steps(values):
    """
    Return a composable function that counts the number of steps.
    The step count is stored in the "_step" key.
    """

    # Find the current step
    step = values.get("_step", -1)

    # Increment the step
    step += 1

    return {**values, "_step": step}


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


def merge_logs(logs, mode):
    """
    Merge a list of logs into a single dictionary.
    The values on each step can be either concatenated or stacked.

    Parameters:
    ----------
    logs: list of dict
        The list of logs to merge.
    mode: str
        The mode to use. Can be "concat" or "stack".
    """

    # Define the function to use for merging
    if mode == "concat":
        fn = jnp.concatenate
    elif mode == "stack":
        fn = jnp.stack
    else:
        raise ValueError(f"Unknown mode {mode}")

    def merge_fn(ls):
        if (isinstance(ls[0], np.ndarray) or isinstance(ls[0], jnp.ndarray)) and len(
            ls[0].shape
        ) > 0:
            return fn(ls)
        else:
            return ls

    # Merge the logs
    merged = {
        key: merge_fn([log[key] for log in logs if key in log])
        for key in set(logs[0]) | set(logs[-1])
    }

    return merged


def mean_keep_dtype(x):
    """
    Compute the mean of an array, without changing its dtype.
    """
    return x.mean(dtype=x.dtype)


def track_progress(keys, every=1, total=None):
    """
    Return a composable function that updates a progress bar with the
    running average of the given keys found in the logs.

    Parameters:
    ----------
    keys: list of str
        The keys to track.
    every: int
        The number of steps between each update of the progress bar.
    total: int or None
        The total number of steps. If None, the progress bar will not be closed.
    """

    @Composable
    def _track_progress(values):
        step = values["_step"]
        logs = values["_logs"]

        # Get the progress bar, or create a new one
        tqdm_bar = values.get("_tqdm", None)
        if tqdm_bar is None:
            tqdm_bar = tqdm(total=total, ncols=140)

        # Update the progress bar, if needed
        if step % every == 0 or (step + 1) % total == 0:
            # Merge all logs so far
            merged_logs = merge_logs(logs, mode="concat")

            # Keep only the keys we want to track
            merged_logs = {k: v for k, v in merged_logs.items() if k in keys}

            # Compute the mean of each key
            mean_logs = jax.tree_util.tree_map(mean_keep_dtype, merged_logs)

            # Format the values
            formatted_logs = jax.tree_util.tree_map(format_digits, mean_logs)

            # Update the description
            desc = " █ ".join(
                [f"{k} {formatted_logs[k]}" for k in keys if k in formatted_logs]
            )
            tqdm_bar.set_description(desc)

        # Update the progress bar
        tqdm_bar.update()

        # Close the progress bar, if needed
        if (step + 1) % total == 0:
            tqdm_bar.close()
            if "_tqdm" in values:
                values.pop("_tqdm")
            return values

        return {**values, "_tqdm": tqdm_bar}

    return _track_progress


def mean_over_epoch(keys):
    """
    Compute the mean of the given metrics over the current epoch.
    """

    @Composable
    def _mean_over_epoch(values):
        epoch_logs = values["_epoch_logs"]
        mean_epoch_logs = {
            "mean_" + k: v.mean(-1) for k, v in epoch_logs.items() if k in keys
        }
        return {**values, "_epoch_logs": {**epoch_logs, **mean_epoch_logs}}

    return _mean_over_epoch


@Composable
def stack_epoch_logs(values):
    """
    Add the logs for the current epoch to the logs of the previous epochs.
    """

    # Get the logs for the current epoch
    logs = values.pop("_logs")
    epoch_logs = merge_logs([merge_logs(logs, "concat")], "stack")

    # Get the logs for the previous epochs
    if "_epoch_logs" in values:
        epoch_logs = merge_logs([values["_epoch_logs"], epoch_logs], "concat")

    return {**values, "_epoch_logs": epoch_logs}


@Composable
@settings.settings_fn
def save_logs(values, *, name, save_log, save_settings):
    """
    Save the logs to a file.
    """

    if save_log:
        with open(f"logs/{name}.pkl", "wb") as f:
            pickle.dump(values["_epoch_logs"], f)

    return values


@Composable
@settings.settings_fn
def save_params(
    values, *, name, save_weights, early_stopping, optimizing_metric, optimizing_mode
):
    """
    Save the parameters of the model to a file.
    """

    if not save_weights:
        return values

    # Get the model parameters
    params = values["params"]
    fixed_params = values["fixed_params"]

    # Merge the parameters
    params = hk.data_structures.merge(params, fixed_params)

    save = not early_stopping

    if early_stopping:
        metric = values["_epoch_logs"][optimizing_metric]

        if optimizing_mode == "min":
            metric = -metric

        if metric.ndim != 1:
            raise ValueError(
                f"Metric {optimizing_metric} should be be a mean over the epoch."
            )

        if metric[-1] == metric.max():
            save = True

    if save:
        with open(f"params/{name}.pkl", "wb") as f:
            pickle.dump(params, f)

    return values
