from .composition import Composable
from tqdm import tqdm
from jax import numpy as jnp
import jax
from tqdm import tqdm
from itertools import chain


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

    # Get the number of digits in the integer part
    integer_len = len(str(int(val)))

    if val.dtype in [jnp.int32, jnp.int64]:
        # If an integer, pad with spaces
        return f"{val:{digits}d}"
    else:
        # If a float, assign all remaining digits to the fractional part
        return f"{val:.{digits - integer_len - 1}f}"


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

    # Merge the logs
    merged = {
        key: fn([log[key] for log in logs if key in log])
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
        if step % every == 0 or step + 1 == total:

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
        if step + 1 == total:
            tqdm_bar.close()

        return {**values, "_tqdm": tqdm_bar}

    return _track_progress
