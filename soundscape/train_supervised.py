import hydra
import jax
import optax
from jax import numpy as jnp

from soundscape import training


def get_optimizer():
    return optax.sgd(1e-3)


def _train_for_epoch(train_fn, call_fn, dataset, batch_size):
    def _train_for_epoch(rng):
        state, epoch_train_results = training.scan_dataset(
            train_fn,
            state,
            dataset.iterate(rng, "train", batch_size),
        )

        _, epoch_eval_results = training.scan_dataset(
            call_fn,
            state,
            dataset.iterate(rng, "val", batch_size),
        )
