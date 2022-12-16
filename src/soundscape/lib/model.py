import haiku as hk
from jax import numpy as jnp
import jax
from oryx.core import sow
import jax_resnet
from flax import linen as nn
import flax
from collections import namedtuple

from . import sow_transforms

def partition(apply_fn, partition_fn, params):
    """
    Partition a model's parameters into a trainable part and a fixed part.
    """

    params, fixed_params = hk.data_structures.partition(partition_fn, params)

    def _apply_fn(params, fixed_params, *args, **kwargs):
        params = hk.data_structures.merge(params, fixed_params)
        return apply_fn(params, *args, **kwargs)

    return _apply_fn, params, fixed_params


def flax2haiku(flax_apply_fn, variables):
    """
    Convert a Flax model's apply function to one
    with a haiku-compatible signature.
    """

    def _apply_fn(params, state, rng, x, is_training=True):
        variables = {"params": params, "batch_stats": state}
        outs, new_variables = flax_apply_fn(
            variables, x, mutable=["batch_stats"] if is_training else []
        )

        new_state = new_variables["batch_stats"] if is_training else state

        return outs, new_state

    return _apply_fn, variables["params"], variables["batch_stats"]


partition_fns = {
    "all": lambda m, n, p: True,
    "none": lambda m, n, p: False,
    "head": lambda m, n, p: "logits" in m,
}


class AddLogits(nn.Module):
    """
    Add a linear layer to the end of a model.
    """

    layers: list
    logits: callable

    @nn.compact
    def __call__(self, *args, **kwargs):
        x = args[0]
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.logits(x)


def resnet(settings, rng, *args, **kwargs):
    """
    Create a ResNet's parameters and apply function.
    """

    size = int(settings["model"]["name"].split("_")[-1])

    # Get the parameters and apply function of a ResNet
    if settings["model"]["weights"] == "imagenet":
        ResNet, variables = jax_resnet.pretrained_resnet(size)
        variables = jax_resnet.common.slice_variables(variables, end=-1)
    else:
        ResNet = lambda: getattr(jax_resnet.resnet, f"ResNet{size}")(n_classes=1)
        variables = ResNet().init(rng, jnp.zeros((1, 224, 224, 3)))

    # Create a linear classifier
    classifier = nn.Dense(settings["data"]["num_classes"], name="logits")
    classifier_params = classifier.init(rng, jnp.zeros((1, 2048)))

    # Replace the last layer of the ResNet with the classifier
    variables = flax.core.unfreeze(variables)
    variables["params"]["logits"] = classifier_params["params"]
    variables = flax.core.freeze(variables)

    net = AddLogits(layers=ResNet().layers[:-1], logits=classifier)

    # Convert the Flax apply function to a haiku-compatible one
    apply_fn, params, state = flax2haiku(net.apply, variables)

    # Partition the parameters into a trainable part and a fixed part
    partition_fn = partition_fns[settings["train"]["partition_fn"]]
    apply_fn, params, fixed_params = partition(apply_fn, partition_fn, params)

    # Sow the outputs of the apply function
    apply_fn = sow_transforms.tuple_to_sow(
        apply_fn, names=["logits", "state"], tag="output"
    )

    return (params, fixed_params, state), apply_fn
