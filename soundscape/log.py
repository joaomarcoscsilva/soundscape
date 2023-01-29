from .composition import Composable
from tqdm import tqdm
from jax import numpy as jnp


def track(keys):
    @Composable
    def _track(values):

        if "logs" in values:
            old_logs = values["logs"]
            to_log = {key: values[key] for key in keys}
            new_logs = {
                key: jnp.concatenate([old_logs[key], to_log[key]]) for key in keys
            }

        else:
            new_logs = {key: values[key] for key in keys}

        return {**values, "logs": new_logs}

    return _track


def log_tqdm(keys, num_steps):
    @Composable
    def _log_tqdm(values):
        bar = values["tqdm"] if "tqdm" in values else tqdm(num_steps)

        tqdm_str = ", ".join([f'{k}: {values["logs"][k].mean(0):.3f}' for k in keys])

        bar.set_description(tqdm_str)
        bar.update()

        return {**values, "tqdm": bar}

    return _log_tqdm
