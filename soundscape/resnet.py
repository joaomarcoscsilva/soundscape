import flax
import haiku as hk
import jax_resnet
from flax import linen as nn
from jax import numpy as jnp

from .composition import Composable
from .settings import settings_fn

# A dictionary of functions that return a boolean indicating whether a given
# parameter should be trained or not.
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
def resnet(
    rng,
    return_values=True,
    *,
    model_name,
    initialization,
    num_classes,
    trainable_weights,
    always_mutable_bn,
):
    """
    Return a resnet composable function that produces logits and batch stats.
    Also return a dictionary with the initial variables.

    Parameters:
    ----------
    rng: jax.random.PRNGKey
        The random number generator key.

    Settings:
    --------
    model_name: str
        The name of the resnet model to use.
    initialization: str
        The initialization scheme to use. Can be "imagenet" or "random".
    num_classes: int
        The number of classes to predict.
    trainable_weights: str
        The partition of the weights to train. Can be "all", "none" or "head".
    always_mutable_bn: bool
        Whether to always use mutable batch statistics.
    """

    # Get the number of layers in the model
    num_layers = int(model_name.split("_")[-1])

    # Initialize the model
    if initialization == "imagenet":
        ResNet, variables = jax_resnet.pretrained_resnet(num_layers)
        variables = jax_resnet.common.slice_variables(variables, end=-1)
    else:
        ResNet = lambda: getattr(jax_resnet.resnet, f"ResNet{num_layers}")(n_classes=1)
        variables = ResNet().init(rng, jnp.zeros((1, 224, 224, 3)))
        variables = jax_resnet.common.slice_variables(variables, end=-1)

    # Create a classifier head
    classifier = nn.Dense(num_classes, name="logits")

    # Find the shape of the last layer, which defines the input to the classifier
    num_layer_groups = len(variables["params"].keys())
    last_layer_group = variables["params"][f"layers_{num_layer_groups}"]
    num_layer_blocks = len(last_layer_group.keys())
    last_layer_block = last_layer_group[f"ConvBlock_{num_layer_blocks-1 }"]
    last_layer_shape = last_layer_block["Conv_0"]["kernel"].shape

    # Initialize the classifier head
    classifier_params = classifier.init(rng, jnp.zeros((1, last_layer_shape[-1])))

    # Add the classifier head's parameters to the model's parameters
    variables = flax.core.unfreeze(variables)
    variables["params"]["logits"] = classifier_params["params"]
    variables = flax.core.freeze(variables)

    # Create the model by merging the backbone and the classifier head
    net = AddLogits(layers=ResNet().layers[:-1], logits=classifier)

    params = variables["params"]
    state = variables["batch_stats"]

    # Partition the parameters into trainable and fixed
    params, fixed_params = hk.data_structures.partition(
        partition_fns[trainable_weights], params
    )

    @Composable
    def call_resnet(values):
        """
        Resnet call function.

        Parameters:
        ----------
        values["params"]: dict
            The trainable parameters of the model.
        values["fixed_params"]: dict
            The fixed parameters of the model.
        values["state"]: dict
            The batch statistics of the model.
        values["inputs"]: jnp.ndarray
            The inputs to the model.
        values["is_training"]: bool
            Whether the model is in training mode.
        """

        inputs = values["inputs"]
        params = values["params"]
        fixed_params = values["fixed_params"]
        state = values["state"]

        # Merge the trainable and fixed parameters
        params = hk.data_structures.merge(params, fixed_params)

        # Normalize the inputs
        inputs = inputs - jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        inputs = inputs / jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))

        # Apply the model
        logits, variables = net.apply(
            {"params": params, "batch_stats": state},
            inputs,
            mutable=["batch_stats"]
            if values["is_training"] or always_mutable_bn
            else [],
        )

        # Update the batch statistics if in training mode
        if values["is_training"]:
            state = variables["batch_stats"]

        return {**values, "logits": logits, "state": state}

    return call_resnet, {"params": params, "fixed_params": fixed_params, "state": state}
