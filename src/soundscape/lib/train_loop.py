from jax import numpy as jnp
from tqdm.auto import tqdm
from functools import partial
import jax
from oryx.core.interpreters.harvest import call_and_reap
from dvclive import Live

from ..data import dataset, dataset_functions as dsfn
from . import utils, log


def append_fn(state, new_aux):
    """
    Append the new values in new_aux to state
    """

    if state is None:
        return {k: [new_aux[k]] for k in new_aux.keys()}

    return {k: state[k] + [new_aux[k]] for k in new_aux.keys()}


def train_fn(settings, update_fn):
    """
    Create a training function to be used with scan_ds.
    """

    @partial(jax.jit, donate_argnums=(1, 2, 4))
    def next_fn(batch, rng, optim_state, params, fixed_params, state):
        rng_batch, rng = jax.random.split(rng)

        batch = dsfn.prepare_image(settings)(batch)

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


def eval_fn(settings, loss_fn):
    """
    Create an inference function to be used with scan_ds.
    """

    @jax.jit
    def next_fn(batch, rng, params, fixed_params, state):
        rng_batch, rng = jax.random.split(rng)

        batch = dsfn.prepare_image(settings)(batch)

        _, aux = loss_fn(
            params,
            fixed_params,
            state,
            rng_batch,
            batch["spec"],
            is_training=True,
            labels=batch["labels"],
        )

        return (rng, params, fixed_params, state), aux

    return next_fn


def scan_ds(
    ds,
    *args,
    next_fn,
    batch_log_fn=None,
    desc_fn=None,
    epoch_log_fn=None,
    log_state=None,
):
    """
    Iterate through a dataset, applying next_fn to each batch.
    The state is accumulated using log_fn, and a description in the progress bar
    is created using desc_fn.
    """

    if desc_fn is not None:
        ds = tqdm(ds)

    log_state = None

    for batch in dataset.jax_dataset(ds):

        jax_batch = utils.dict_filter(lambda x: isinstance(x, jnp.ndarray), batch)

        args, aux = next_fn(jax_batch, *args)
        log_state = append_fn(log_state, aux)

        if batch_log_fn is not None:
            batch_log_fn(log_state)

        if desc_fn is not None:
            ds.set_description(desc_fn(log_state, aux))

    if epoch_log_fn is not None:
        epoch_log_fn(log_state, args)

    return log_state, *args


def get_train_epoch_fn(train_fn, epoch, logger=None):
    """
    Train on a dataset.
    """

    return lambda ds, *args: scan_ds(
        ds,
        *args,
        next_fn=train_fn,
        batch_log_fn=log.batch_log_fn(logger),
        desc_fn=log.running_average_desc_fn(f"Epoch {epoch:04d} "),
        epoch_log_fn=None,
    )


def get_eval_epoch_fn(eval_fn, logger=None, prefix='eval_'):
    """
    Evaluate on a dataset.
    """

    return lambda ds, *args: scan_ds(
        ds,
        *args,
        next_fn=eval_fn,
        batch_log_fn=None,
        desc_fn=log.running_average_desc_fn("Evaluation "),
        epoch_log_fn=log.epoch_log_fn(logger, prefix=prefix),
    )
