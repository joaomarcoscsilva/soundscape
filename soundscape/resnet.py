from .composition import Composable
from .settings import from_file
import jax_resnet
from flax import linen as nn
import flax
from jax import numpy as jnp
import haiku as hk

settings_fn, settings_dict = from_file("model_settings.yaml")


# Partitioning functions to define which parameters are trainable
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
    def __call__(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.logits(x)


@settings_fn
def resnet(rng, *, model_name, initialization, num_classes, trainable_weights):
    """
    Return a resnet composable function that produces logits and batch stats.
    Also return a dictionary with the initial variables.
    """

    num_layers = int(model_name.split("_")[-1])

    if initialization == "imagenet":
        ResNet, variables = jax_resnet.pretrained_resnet(num_layers)
        variables = jax_resnet.common.slice_variables(variables, end=-1)
    else:
        ResNet = lambda: getattr(jax_resnet.resnet, f"ResNet{num_layers}")(n_classes=1)
        variables = ResNet().init(rng, jnp.zeros((1, 224, 224, 3)))

    classifier = nn.Dense(num_classes, name="logits")

    num_layer_groups = len(variables["params"].keys())
    last_layer_group = variables["params"][f"layers_{num_layer_groups}"]
    num_layer_blocks = len(last_layer_group.keys())
    last_layer_block = last_layer_group[f"ConvBlock_{num_layer_blocks-1 }"]
    last_layer_shape = last_layer_block["Conv_0"]["kernel"].shape

    classifier_params = classifier.init(rng, jnp.zeros((1, last_layer_shape[-1])))

    variables = flax.core.unfreeze(variables)
    variables["params"]["logits"] = classifier_params["params"]
    variables = flax.core.freeze(variables)

    net = AddLogits(layers=ResNet().layers[:-1], logits=classifier)

    params = variables["params"]
    state = variables["batch_stats"]

    params, fixed_params = hk.data_structures.partition(
        partition_fns[trainable_weights], params
    )

    @Composable
    def call_resnet(values):
        params = values["params"]
        fixed_params = values["fixed_params"]
        state = values["state"]

        params = hk.data_structures.merge(params, fixed_params)

        logits, variables = net.apply(
            {"params": params, "batch_stats": state},
            values["inputs"],
            mutable=["batch_stats"] if values["is_training"] else [],
        )

        return {**values, "logits": logits, "state": variables["batch_stats"]}

    return call_resnet, {"params": params, "fixed_params": fixed_params, "state": state}
