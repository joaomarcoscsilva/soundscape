from oryx.core.interpreters.harvest import call_and_reap
import optax
import jax
import json, pickle
from IPython import embed


from ..data import dataset, dataset_functions as dsfn
from .settings import SettingsFunction, settings
from . import loss_transforms, sow_transforms, model, train_loop


@SettingsFunction
def get_optimizer(settings, params):
    """
    Create an optimizer.
    """

    steps = (
        dsfn.num_events("train_labels.csv")
        * settings["train"]["num_epochs"]
        / settings["train"]["batch_size"]
    )

    schedule = optax.cosine_decay_schedule(
        init_value=settings["train"]["learning_rate"], decay_steps=steps
    )

    optim = optax.adamw(learning_rate=schedule)

    optim_state = optim.init(params)

    return optim, optim_state


@SettingsFunction
def get_loss(settings, apply_fn):
    """
    Create a loss function.
    """

    class_weights = 1 / (
        dsfn.class_frequencies("train_labels.csv") * settings["data"]["num_classes"]
    )

    loss_fn = optax.softmax_cross_entropy
    bal_loss_fn = loss_transforms.weighted(loss_fn, class_weights=class_weights)

    acc_fn = lambda logits, labels: logits.argmax(axis=-1) == labels.argmax(axis=-1)
    bal_acc_fn = loss_transforms.weighted(acc_fn, class_weights)

    pred_fn = lambda logits, labels: logits

    loss_fn = loss_transforms.mean_loss(loss_fn)
    bal_loss_fn = loss_transforms.mean_loss(bal_loss_fn)
    acc_fn = loss_transforms.mean_loss(acc_fn)
    bal_acc_fn = loss_transforms.mean_loss(bal_acc_fn)

    if settings["train"]["balanced"]:
        loss_fn = sow_transforms.sow_fns(
            [bal_loss_fn, loss_fn, bal_acc_fn, acc_fn, pred_fn],
            names=[
                "balanced_loss",
                "loss",
                "balanced_accuracy",
                "accuracy",
                "predictions",
            ],
            tag="model",
        )

    else:
        loss_fn = sow_transforms.sow_fns(
            [loss_fn, bal_loss_fn, acc_fn, bal_acc_fn, pred_fn],
            names=[
                "loss",
                "balanced_loss",
                "accuracy",
                "balanced_accuracy",
                "predictions",
            ],
            tag="model",
        )

    # Create applied loss
    loss_fn = loss_transforms.applied_loss(loss_fn, apply_fn)
    loss_fn = jax.jit(loss_fn, static_argnames=("is_training",))

    return loss_fn


@SettingsFunction
def get_update(settings, loss_fn, optim):
    """
    Create an update function.
    """
    update_fn = loss_transforms.update(loss_fn, optim)
    update_fn = sow_transforms.sow_to_tuple(update_fn, name="state", tag="output")
    update_fn = jax.jit(update_fn)

    return update_fn


def log_to_disk(train_logs, val_logs, best, save_on_best):

    metrics = {"train": train_logs, "val": val_logs}
    opt_metric = val_logs[-1][settings["train"]["optimizing_metric"]].mean()

    with open(f"metrics.json", "w") as f:
        json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), metrics), f, indent=4)

    if opt_metric > best:
        best = opt_metric
        with open(f"models/model.pkl", "wb") as f:
            pickle.dump(save_on_best, f)

    return best


def train(rng, train_ds, val_ds=None, model_fn=model.resnet):
    model_rng, data_rng, rng = jax.random.split(rng, 3)

    (params, fixed_params, state), apply_fn = model_fn(model_rng)
    optim, optim_state = get_optimizer(params)

    loss_fn = get_loss(apply_fn)
    update_fn = get_update(loss_fn, optim)

    metrics = ["loss", "balanced_loss", "accuracy", "balanced_accuracy"]
    update_fn = call_and_reap(update_fn, tag="model", allowlist=metrics)
    loss_fn = call_and_reap(loss_fn, tag="model", allowlist=metrics + ["predictions"])

    train_fn = train_loop.train_fn(update_fn)
    eval_fn = train_loop.eval_fn(loss_fn)

    best = -1

    train_logs = []
    eval_logs = []

    for e in range(settings["train"]["num_epochs"]):

        rng, rng_shuffle, rng_frag, rng_val = jax.random.split(rng, 4)

        epoch_ds = train_ds.shuffle(buffer_size=1000, seed=rng_shuffle[0])
        epoch_ds = dataset.fragment_dataset(epoch_ds, rng_frag)
        epoch_ds = epoch_ds.batch(settings["train"]["batch_size"])
        epoch_ds = epoch_ds.prefetch(4)

        train_epoch = train_loop.get_train_epoch_fn(train_fn, e)
        train_out = train_epoch(epoch_ds, rng, optim_state, params, fixed_params, state)
        train_log, rng, optim_state, params, fixed_params, state = train_out
        train_logs.append(train_log)

        if val_ds is None:
            continue

        epoch_val_ds = dataset.fragment_dataset(val_ds, rng_val)
        epoch_val_ds = epoch_val_ds.batch(settings["train"]["batch_size"])
        epoch_val_ds = epoch_val_ds.prefetch(4)

        eval_epoch = train_loop.get_eval_epoch_fn(eval_fn, desc_keys=metrics)
        eval_out = eval_epoch(epoch_val_ds, rng, params, fixed_params, state)
        eval_log, rng, *_ = eval_out
        eval_logs.append(eval_log)

        best = log_to_disk(train_logs, eval_logs, best, (params, fixed_params, state))

        print("")
