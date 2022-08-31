from functools import partial, wraps
import optax
import jax
import equinox as eqx


def weighted_loss(loss_fn, class_weights=None):
    """
    Takes in a loss function and returns a new function that
    weights samples according to their target labels. The weights for each
    class must be passed as an argument to the constructor.
    """

    if class_weights == None:
        return loss_fn

    @wraps(loss_fn)
    def weighted_loss_fn(logits, labels):
        weights = class_weights[labels.argmax(axis=-1)]
        return weights * loss_fn(logits=logits, labels=labels)

    return weighted_loss_fn


def applied_loss(loss_fn):
    """
    Takes in a loss function and returns a new function that
    takes in an equinox model, inputs, labels and a key and
    returns the loss value and a dict with the logits.
    """

    @wraps(loss_fn)
    def applied_loss_fn(model, inputs, labels, key):
        logits = jax.vmap(model, axis_name="batch")(inputs, key=key)
        loss = loss_fn(logits=logits, labels=labels).mean()
        return loss, logits

    return applied_loss_fn


def update_from_loss(value_and_grad_loss_fn):
    """
    Takes in a loss function transformed by jax.value_and_grad
    with hax_aux = True and returns an update function.
    """

    # @wraps(value_and_grad_loss_fn)
    def update_from_loss_fn(model, inputs, labels, optim, optim_state, key):
        ((loss, aux), grad) = value_and_grad_loss_fn(model, inputs, labels, key=key)
        updates, optim_state = optim.update(grad, optim_state)
        model = eqx.apply_updates(model, updates)
        return model, optim_state, loss, aux

    return update_from_loss_fn
