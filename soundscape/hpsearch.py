import jax
import jax.numpy as jnp
import itertools
import pickle
import pprint
import os

from soundscape import supervised, log, settings
from soundscape.composition import hashable_dict, Composable


@Composable
@settings.settings_fn
def grid_search(values, *, hpsearch_ranges):
    exp_index = values["_step"]

    if not all(isinstance(v, dict) for v in hpsearch_ranges.values()):
        raise ValueError("grid_search only supports sets")

    hps = itertools.product(*hpsearch_ranges.values())
    changed_settings = list(hps)[exp_index]
    changed_settings = dict(zip(hpsearch_ranges.keys(), changed_settings))

    return {**values, "changed_settings": changed_settings}


@Composable
@settings.settings_fn
def uniform_search(values, *, hpsearch_ranges):
    rng = values["rng"]

    sampled = {}

    for k, v in hpsearch_ranges.items():
        rng, _rng = jax.random.split(rng)
        if isinstance(v, list):
            sampled[k] = jax.random.uniform(_rng, minval=v[0], maxval=v[1])
        elif isinstance(v, dict):
            idx = jax.random.choice(_rng, len(v))
            sampled[k] = list(v.keys())[idx]

    return {**values, "changed_settings": sampled, "rng": rng}


hp_sample_functions = {"grid": grid_search, "uniform": uniform_search}


@Composable
def run_experiment(values):
    changed_settings = values["changed_settings"]

    name = settings.settings_dict()["name"]
    name = f"{name}[{values['_step']}]"

    print("\nSettings:")
    pprint.pprint(changed_settings, width=1)

    with settings.Settings({**changed_settings, "name": name}):
        logs = supervised.train()

    return {**values, "logs": logs}


@Composable
@settings.settings_fn
def get_best_epoch(values, *, optimizing_metric, optimizing_mode):
    sign = 1 if optimizing_mode == "max" else -1
    metric = values["logs"][optimizing_metric]
    best_epoch = (sign * metric).argmax()
    return {**values, "metric": metric[best_epoch], "best_epoch": best_epoch}


@settings.settings_fn
def run_search(
    *,
    hpsearch_iterations,
    hpsearch_function,
    hpsearch_seed,
    name,
    hpsearch_indices,
    hpsearch_begin,
):
    if os.path.exists(f"logs/{name}.pkl"):
        raise ValueError(f"logs/{name}.pkl already exists")

    search_fn = log.count_steps | hp_sample_functions[hpsearch_function]
    run_fn = run_experiment | get_best_epoch

    rng = jax.random.PRNGKey(hpsearch_seed)

    values = {"rng": rng}

    results = []

    for i in range(hpsearch_iterations):
        values = search_fn(values)

        if i < hpsearch_begin:
            continue

        if len(hpsearch_indices) > 0 and i not in hpsearch_indices:
            continue

        values = run_fn(values)

        results.append(values)

        with open(f"logs/{name}.pkl", "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    with settings.Settings.from_command_line():
        run_search()
