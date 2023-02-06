from .composition import Composable
from tqdm import tqdm
from jax import numpy as jnp
import jax
from tqdm import tqdm
from itertools import chain


def track(keys, prefix=""):
    """
    Track the values of the given keys and store them in the logs.
    The logs are stored as a list in the "_logs" key.
    """

    @Composable
    def _track(values):

        step_logs = {prefix + key: values[key] for key in keys}
        logs = values.get("_logs", [])
        logs.append(step_logs)

        return {**values, "_logs": logs}

    return _track


def count_steps(values):
    step = values.get("_step", None)
    step = values["_step"] if "_step" in values else 0
    return {**values, "_step": step + 1}


def format_digits(val, digits=6):
    integer_len = len(str(int(val)))
    if val.dtype in [jnp.int32, jnp.int64]:
        return f"{val:{digits}d}"
    else:
        return f"{val:.{digits - integer_len - 1}f}"


def merge_logs(logs, mode):

    if mode == "concat":
        fn = jnp.concatenate
    elif mode == "stack":
        fn = jnp.stack
    else:
        raise ValueError(f"Unknown mode {mode}")

    merged = {key: fn([log[key] for log in logs]) for key in logs[0]}

    return merged


def mean_keep_dtype(x):
    return x.mean(dtype=x.dtype)


def track_progress(keys, every=1, total=None):
    @Composable
    def _track_progress(values):

        step = values["_step"]
        logs = values["_logs"]

        tqdm_bar = values.get("_tqdm", None)
        if tqdm_bar is None:
            tqdm_bar = tqdm(total=total, ncols=140)

        if step % every == 0 or step + 1 == total:
            merged_logs = merge_logs(logs, mode="concat")
            merged_logs = {k: v for k, v in merged_logs.items() if k in keys}
            mean_logs = jax.tree_util.tree_map(mean_keep_dtype, merged_logs)
            formatted_logs = jax.tree_util.tree_map(format_digits, mean_logs)

            desc = " █ ".join([f"{k} {v}" for k, v in formatted_logs.items()])
            tqdm_bar.set_description(desc)

            if step + 1 == total:
                tqdm_bar.close()

        tqdm_bar.update()

        return {**values, "_tqdm": tqdm_bar}

    return _track_progress
