from oryx.core.interpreters.harvest import call_and_reap
import optax
import jax
from IPython import embed

from ..data import dataset, dataset_functions as dsfn
from . import loss_transforms, sow_transforms, model, train_loop, log


def get_optimizer(settings, params):
    """
    Create an optimizer.
    """

    steps = (
        dsfn.num_events(settings)("train_labels.csv")
        * settings["train"]["num_epochs"]
        / settings["train"]["batch_size"]
    )

    schedule = optax.cosine_decay_schedule(
        init_value=settings["train"]["learning_rate"], decay_steps=steps
    )

    optim = optax.adamw(learning_rate=schedule)

    optim_state = optim.init(params)

    return optim, optim_state


def get_loss(settings, apply_fn):
    """
    Create a loss function.
    """

    class_weights = 1 / (
        dsfn.class_frequencies(settings)("train_labels.csv")
        * settings["data"]["num_classes"]
    )

    loss_fn = optax.softmax_cross_entropy
    bal_loss_fn = loss_transforms.weighted(loss_fn, class_weights=class_weights)

    acc_fn = lambda logits, labels: logits.argmax(axis=-1) == labels.argmax(axis=-1)
    bal_acc_fn = loss_transforms.weighted(acc_fn, class_weights)

    logit_fn = lambda logits, labels: logits
    pred_fn = lambda logits, labels: logits.argmax(axis=-1)
    label_fn = lambda logits, labels: labels.argmax(axis=-1)
    prob_fn = lambda logits, labels: (jax.nn.softmax(logits) * labels).sum(axis=-1)

    # loss_fn = loss_transforms.mean_loss(loss_fn)
    # bal_loss_fn = loss_transforms.mean_loss(bal_loss_fn)
    # acc_fn = loss_transforms.mean_loss(acc_fn)
    # bal_acc_fn = loss_transforms.mean_loss(bal_acc_fn)

    main_loss_fn = loss_transforms.mean_loss(
        bal_loss_fn if settings["train"]["balanced"] else loss_fn
    )

    loss_fn = sow_transforms.sow_fns(
        [
            main_loss_fn,
            loss_fn,
            bal_loss_fn,
            acc_fn,
            bal_acc_fn,
            logit_fn,
            pred_fn,
            label_fn,
            prob_fn,
        ],
        names=[
            "main_loss",
            "loss",
            "balanced_loss",
            "accuracy",
            "balanced_accuracy",
            "logits",
            "predictions",
            "labels",
            "probabilities",
        ],
        tag="model",
    )

    # Create applied loss
    loss_fn = loss_transforms.applied_loss(loss_fn, apply_fn)
    loss_fn = jax.jit(loss_fn, static_argnames=("is_training",))

    return loss_fn


def get_update(settings, loss_fn, optim):
    """
    Create an update function.
    """

    update_fn = loss_transforms.update(loss_fn, optim)
    update_fn = sow_transforms.sow_to_tuple(update_fn, name="state", tag="output")
    update_fn = jax.jit(update_fn)

    return update_fn


def train(settings, rng, train_ds, val_ds=None, model_fn=model.resnet):
    """
    Train a model.
    """

    model_rng, data_rng, rng = jax.random.split(rng, 3)

    (params, fixed_params, state), apply_fn = model_fn(settings, model_rng)
    optim, optim_state = get_optimizer(settings, params)

    loss_fn = get_loss(settings, apply_fn)
    update_fn = get_update(settings, loss_fn, optim)

    metrics = ["loss", "balanced_loss", "accuracy", "balanced_accuracy"]
    update_fn = call_and_reap(update_fn, tag="model", allowlist=metrics)
    loss_fn = call_and_reap(
        loss_fn,
        tag="model",
        allowlist=metrics + ["labels", "predictions", "probabilities"],
    )

    train_fn = train_loop.train_fn(settings, update_fn)
    eval_fn = train_loop.eval_fn(settings, loss_fn)

    logger = log.Logger(settings["train"]["optimizing_metric"])

    for e in range(settings["train"]["num_epochs"]):

        rng, rng_shuffle, rng_frag, rng_val = jax.random.split(rng, 4)

        epoch_ds = train_ds.shuffle(buffer_size=1000, seed=rng_shuffle[0])
        epoch_ds = dataset.fragment_dataset(settings, epoch_ds, rng_frag)
        epoch_ds = epoch_ds.batch(settings["train"]["batch_size"])
        epoch_ds = epoch_ds.prefetch(4)

        train_epoch = train_loop.get_train_epoch_fn(train_fn, e, logger)
        train_out = train_epoch(epoch_ds, rng, optim_state, params, fixed_params, state)
        train_log, rng, optim_state, params, fixed_params, state = train_out

        if val_ds is None:
            continue

        epoch_val_ds = dataset.fragment_dataset(settings, val_ds, rng_val)
        epoch_val_ds = epoch_val_ds.shuffle(buffer_size=1000, seed=0)
        epoch_val_ds = epoch_val_ds.batch(settings["train"]["batch_size"])
        epoch_val_ds = epoch_val_ds.prefetch(4)

        eval_epoch = train_loop.get_eval_epoch_fn(eval_fn, logger)
        eval_out = eval_epoch(epoch_val_ds, rng, params, fixed_params, state)
        eval_log, rng, *_ = eval_out

        if settings["train"]["log_train"]:
            epoch_val_train_ds = dataset.fragment_dataset(settings, train_ds, rng_val)
            epoch_val_train_ds = epoch_val_train_ds.shuffle(buffer_size=1000, seed=0)
            epoch_val_train_ds = epoch_val_train_ds.batch(
                settings["train"]["batch_size"]
            )
            epoch_val_train_ds = epoch_val_train_ds.prefetch(4)

            train_eval_epoch = train_loop.get_eval_epoch_fn(
                eval_fn, logger, prefix="train_"
            )
            train_eval_out = train_eval_epoch(
                epoch_val_train_ds, rng, params, fixed_params, state
            )
            train_eval_log, rng, *_ = train_eval_out

        print("")
