from functools import partial
from typing import Callable, NamedTuple

import hydra
import jax
import optax
from jax import random
from omegaconf import DictConfig

from soundscape import log, metrics, optimizing
from soundscape.dataset import dataloading, preprocessing
from soundscape.models import base_model
from soundscape.types import Batch, Predictions


class TrainingEnvironment(NamedTuple):
    dataloader: dataloading.DataLoader
    preprocess: Callable[[Batch, bool], Batch]
    model: base_model.Model
    optimizer: optax.GradientTransformation
    epoch_logger: log.Logger
    logger: log.Logger
    num_epochs: int
    metrics: Callable[[Batch, Predictions], Batch]
    settings: DictConfig


def train_for_epoch(rng, model_state, epoch_i, env):

    rng_train, rng_val, rng_test, rng = random.split(rng, 4)

    env.epoch_logger.restart()
    env.epoch_logger.update({"epoch": epoch_i})

    for batch in env.dataloader.iterate(rng_train, "train"):
        batch = env.preprocess(batch, training=True)
        outputs, model_state = optimizing.update(
            batch, model_state, env.model, env.optimizer
        )
        outputs = env.metrics(batch, outputs)
        env.epoch_logger.update(batch, outputs)

    for batch in env.dataloader.iterate(rng_val, "val"):
        batch = env.preprocess(batch, training=False)
        outputs, _ = env.model(batch, model_state)
        outputs = env.metrics(batch, outputs)
        env.epoch_logger.update(batch, outputs, prefix="val_")

    if env.dataloader.include_test:
        for batch in env.dataloader.iterate(rng_test, "test"):
            batch = env.preprocess(batch, training=False)
            outputs, _ = env.model(batch, model_state)
            outputs = env.metrics(batch, outputs)
            env.epoch_logger.update(batch, outputs, prefix="test_")

    return env.epoch_logger.close(), model_state


def train(rng, model_state, env):
    env.logger.restart()

    for epoch_i in range(env.settings.optimizer.epochs):
        logs, model_state = train_for_epoch(rng, model_state, epoch_i, env)
        env.logger.update(logs)
        if env.logger.early_stop():
            print("\nEarly stopping")
            break

    return env


def instantiate(settings):
    rng = random.PRNGKey(settings.seed)
    rng_dataloader, rng_model, rng = random.split(rng, 3)

    dataloader = dataloading.get_dataloader(rng_dataloader, settings.dataloader)

    preprocess = partial(
        preprocessing.preprocess,
        dataloader=dataloader,
        augs_config=settings.augmentation,
    )

    model, model_state = base_model.get_model(
        rng_model, dataloader.num_classes, settings.model
    )

    optimizer, model_state = optimizing.get_optimizer(
        model_state, settings.optimizer, dataloader.get_steps_per_epoch()
    )

    metrics_fn = metrics.get_metrics_function(dataloader.prior_weights())

    logger = log.get_logger(
        settings.logger, pbar_len=settings.optimizer.epochs, pbar_level=1
    )

    epoch_logger = log.get_logger(settings.epoch_logger, pbar_len=len(dataloader))

    return (
        rng,
        model_state,
        TrainingEnvironment(
            dataloader=dataloader,
            preprocess=preprocess,
            model=model,
            optimizer=optimizer,
            epoch_logger=epoch_logger,
            logger=logger,
            num_epochs=settings.optimizer.epochs,
            metrics=metrics_fn,
            settings=settings,
        ),
    )


@hydra.main(
    config_path="../../settings",
    config_name="experiment/linear_leec12.yaml",
    version_base=None,
)
def main(settings):
    rng, model_state, env = instantiate(settings.experiment)
    train(rng, model_state, env)


if __name__ == "__main__":
    main()
