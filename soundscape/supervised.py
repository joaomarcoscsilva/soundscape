from functools import partial
from typing import Callable, NamedTuple

import optax
from jax import random

from soundscape import Batch, base_model, log, optimizing
from soundscape.dataset import dataloading, preprocessing

# logged_keys = [
#     "ce_loss",
#     "brier",
#     "acc",
#     "acc_nb",
#     "bal_acc",
#     "bal_acc_nb",
#     "epoch",
#     "id",
#     "labels",
#     "logits",
#     "one_hot_labels",
# ]
# pbar_keys = [
#     "epoch",
#     "ce_loss",
#     "bal_acc",
#     "val_ce_loss",
#     "val_bal_acc",
#     "val_acc_nb",
# ]


class TrainingEnvironment(NamedTuple):
    dataloader: dataloading.DataLoader
    preprocess: Callable[[Batch, bool], Batch]
    model: base_model.Model
    optimizer: optax.GradientTransformation
    epoch_logger: log.Logger
    logger: log.Logger
    num_epochs: int


def train_for_epoch(rng, model_state, epoch_i, env):

    rng_train, rng_val, rng_test, rng = random.split(rng, 4)

    env.epoch_logger.update({"epoch": epoch_i})

    for batch in env.dataloader.iterate(rng_train, "train"):
        batch = env.preprocess(batch, training=True)
        outputs, model_state = optimizing.update(
            batch, model_state, env.model, env.optimizer
        )
        outputs = env.metrics(batch, outputs)
        env.epoch_logger.update(batch, outputs, prefix="train")

    for batch in env.dataloader.iterate(rng_val, "val"):
        batch = env.preprocess(batch, training=False)
        outputs, _ = env.model(batch, model_state)
        env.epoch_logger.update(batch, outputs, prefix="val")

    if env.dataloader.include_test:
        for batch in env.dataloader.iterate(rng_test, "test"):
            batch = env.preprocess(batch, training=False)
            outputs, _ = env.model(batch, model_state)
            env.epoch_logger.update(batch, outputs, prefix="test")

    return env.epoch_logger.close(), model_state


def train(rng, model_state, env):

    for epoch_i in range(env.num_epochs):
        env.epoch_logger.reset()
        logs, model_state = train_for_epoch(rng, model_state, epoch_i, env)
        env.logger.update(logs)


def instantiate(settings):
    rng = random.PRNGKey(settings.seed)
    rng_dataloader, rng_model, rng = random.split(rng)

    dataloader = dataloading.get_dataloader(rng_dataloader, settings.dataloader)

    preprocess = partial(
        preprocessing.preprocess,
        dataloader=dataloader,
        augs_config=settings.augmentation,
    )

    model, model_state = base_model.get_model(
        rng_model, dataloader.num_classes, settings.model
    )

    optimizer, model_state = optimizing.get_optimizer(model_state, settings.training)

    metrics = metrics.get_metric_fn(settings.dataloader.prior_weights())

    return TrainingEnvironment(
        dataloader=dataloader,
        preprocess=preprocess,
        model=model,
        optimizer=optimizer,
        epoch_logger=log.Logger(),
        logger=log.Logger(),
        num_epochs=settings.epochs,
        metrics=metrics,
    )
