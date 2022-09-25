from jax import numpy as jnp
from tqdm.auto import tqdm
from functools import partial
import jax
from oryx.core.interpreters.harvest import call_and_reap

from ..data import dataset, dataset_functions as dsfn
from . import utils


def at_least_one_dim(x):
    return x if len(x.shape) else x[None]


def accumulate_state(state, new_aux):
    """
    Accumulate the values of new_aux into state.
    """

    if state is None:
        return jax.tree_util.tree_map(at_least_one_dim, new_aux)

    return {
        k: jnp.concatenate((state[k], at_least_one_dim(new_aux[k])))
        for k in state.keys()
        if k != "state"
    }


def train_fn(update_fn):
    """
    Create a training function to be used with scan_ds.
    """

    @partial(jax.jit, donate_argnums=(1, 2, 4))
    def next_fn(batch, rng, optim_state, params, fixed_params, state):
        batch = dsfn.prepare_batch(batch)

        rng_batch, rng = jax.random.split(rng)

        (optim_state, params, state), aux = update_fn(
            optim_state,
            params,
            fixed_params,
            state,
            rng_batch,
            batch["spec"],
            labels=batch["labels"],
        )

        return (rng, optim_state, params, fixed_params, state), aux

    return next_fn


def eval_fn(loss_fn):
    """
    Create an inference function to be used with scan_ds.
    """

    @jax.jit
    def next_fn(batch, rng, params, fixed_params, state):
        batch = dsfn.prepare_batch(batch)

        rng_batch, rng = jax.random.split(rng)

        _, aux = loss_fn(
            params,
            fixed_params,
            state,
            rng_batch,
            batch["spec"],
            is_training=False,
            labels=batch["labels"],
        )

        return (rng, params, fixed_params, state), aux

    return next_fn


def desc_fn(prefix, keys=None):
    """
    Create a description function for the training loop.
    """

    def desc(logs, aux):
        s = prefix
        for k in sorted(keys) if keys is not None else sorted(aux.keys()):
            s += f" {k}: {logs[k].mean():.3f} "
        return s

    return desc


def scan_ds(ds, next_fn, log_fn, desc_fn=None, *args):
    """
    Iterate through a dataset, applying next_fn to each batch.
    The state is accumulated using log_fn, and a description in the progress bar
    is created using desc_fn.
    """

    if desc_fn is not None:
        ds = tqdm(ds)

    logs = None

    for batch in dataset.jax_dataset(ds):
        jax_batch = utils.dict_map(
            lambda x: x if isinstance(x, jnp.ndarray) else None, batch
        )
        args, aux = next_fn(jax_batch, *args)
        logs = log_fn(logs, aux)
        if desc_fn is not None:
            ds.set_description(desc_fn(logs, aux))

    return logs, *args


def get_train_epoch_fn(train_fn, epoch, desc_keys=None):
    """
    Train on a dataset.
    """

    return lambda ds, *args: scan_ds(
        ds, train_fn, accumulate_state, desc_fn(f"Epoch {epoch:04d} ", desc_keys), *args
    )


def get_eval_epoch_fn(eval_fn, desc_keys=None):
    """
    Evaluate on a dataset.
    """

    return lambda ds, *args: scan_ds(
        ds, eval_fn, accumulate_state, desc_fn("Validation ", desc_keys), *args
    )
