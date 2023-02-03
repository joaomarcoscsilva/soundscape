from .composition import Composable
from tqdm import tqdm
from jax import numpy as jnp
import numpy as np
from tqdm import tqdm


def format_digits(val, digits):
    integer_len = len(str(int(val)))
    if val.dtype == jnp.float32:
        return f"{val:.{digits - integer_len - 1}f}"
    else:
        return f"{val:{digits}d}"


def merge(old_logs, new_logs):

    if old_logs is not None:
        for key in new_logs:
            if key in old_logs:
                old_logs[key].extend(new_logs[key])
            else:
                old_logs[key] = new_logs[key]
    else:
        old_logs = new_logs

    return old_logs


def track(keys, prefix=""):
    @Composable
    def _track(values):

        step_logs = {prefix + key: [values[key]] for key in keys}
        old_logs = values.get("_logs", None)
        new_logs = merge(old_logs, step_logs)

        return {**values, "_logs": new_logs}

    return _track


def concatenate_logs(logs, mean=True):
    if mean:
        dtypes = {key: logs[key][0].dtype for key in logs}
        agg_log = {
            key: jnp.concatenate(logs[key]).mean(dtype=dtypes[key]) for key in logs
        }
    else:
        agg_log = {key: np.concatenate(logs[key]) for key in logs}

    return agg_log


def track_progress(keys, every=1, total=None):
    @Composable
    def _track_progress(values):

        tqdm_bar = values.get("_tqdm", None)
        if tqdm_bar is None:
            tqdm_bar = tqdm(total=total, ncols=140)

        step = values.get("_step", None)
        step = values["_step"] if "_step" in values else 0

        logs = values["_logs"]

        if step % every == 0 or step + 1 == total:
            agg_log = concatenate_logs(logs)
            logged_keys = [key for key in keys if key in agg_log]
            tqdm_bar.set_description(
                " █ ".join(
                    [f"{key} {format_digits(agg_log[key], 6)}" for key in logged_keys]
                )
                + " █"
            )

        tqdm_bar.update()

        if step + 1 == total:
            tqdm_bar.close()
            logs = concatenate_logs(logs, mean=False)

        return {**values, "_tqdm": tqdm_bar, "_step": step + 1, "_logs": logs}

    return _track_progress
