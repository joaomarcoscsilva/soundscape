from functools import partial

import jax
import optax


def get_optimizer(model_state, training_settings, dataloader):
    steps_per_epoch = dataloader.get_steps_per_epoch() * training_settings.epochs

    lr_schedule = optax.cosine_decay_schedule(
        init_value=10 ** (training_settings.log_learning_rate),
        decay_steps=steps_per_epoch,
    )

    if training_settings.optim_name == "adamw":
        base_optim_transform = optax.scale_by_adam()
    elif training_settings.optim_name == "sgd":
        if training_settings.sub_log_momentum is not None:
            base_optim_transform = optax.trace(
                decay=1 - 10**training_settings.sub_log_momentum
            )
        else:
            base_optim_transform = optax.identity()
    else:
        raise ValueError(f"Unknown optimizer: {training_settings.optim_name}")

    if training_settings.log_weight_decay is not None:
        weight_decay_transform = optax.add_decayed_weights(
            10**training_settings.log_weight_decay
        )
    else:
        weight_decay_transform = optax.identity()

    optimizer = optax.chain(
        base_optim_transform,
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        weight_decay_transform,
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    optim_state = optimizer.init(model_state.params)

    return optimizer, model_state._replace(optim_state=optim_state)


@partial(jax.jit, static_argnames=["optimizer"])
def _apply_grads(optimizer, model_state, grads):
    updates, optim_state = optimizer.update(
        grads,
        model_state.optim_state,
        model_state.params,
    )

    params = optax.apply_updates(model_state.params, updates)

    return model_state._replace(params=params, optim_state=optim_state)


def update(batch, model_state, model, optimizer):
    outputs, model_state = model(batch, model_state, is_training=True)

    outputs, model_state, grads = model.value_and_grad(
        batch, model_state, is_training=True
    )
    model_state = _apply_grads(optimizer, model_state, grads)
    return outputs, model_state
