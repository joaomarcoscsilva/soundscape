from .composition import Composable
from tqdm import tqdm
from jax import numpy as jnp
from tqdm import tqdm


def track(keys):
    @Composable
    def _track(values):

        if "_logs" in values:
            new_logs = values["_logs"]
            for key in keys:
                new_logs[key].append(values[key])

        else:
            new_logs = {key: [values[key]] for key in keys}

        return {**values, "_logs": new_logs}

    return _track


def track_progress(keys, every=1, total=None):
    @Composable
    def _track_progress(values):

        tqdm_bar = values["_tqdm"] if "_tqdm" in values else tqdm(total=total)
        step = values["_step"] if "_step" in values else 0

        if step % every == 0:
            tqdm_bar.set_postfix(
                {key: jnp.concatenate(values["_logs"][key]).mean() for key in keys}
            )

        tqdm_bar.update()

        return {**values, "_tqdm": tqdm_bar, "_step": step + 1}

    return _track_progress
