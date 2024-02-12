import flax
import haiku as hk
import jax
import jax_resnet
from flax import linen as nn
from jax import numpy as jnp

from ..dataset.dataloading import Batch
from .base_model import Model, ModelState, Predictions, model_creators, partition_fns


class AddLogitsModule(nn.Module):
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


def get_last_layer_shape(variables):
    num_layer_groups = len(variables["params"].keys())
    last_layer_group = variables["params"][f"layers_{num_layer_groups}"]
    num_layer_blocks = len(last_layer_group.keys())
    last_layer_block = last_layer_group[f"ConvBlock_{num_layer_blocks-1 }"]
    last_layer_shape = last_layer_block["Conv_0"]["kernel"].shape
    return last_layer_shape


def _classifier_dummy_input(variables):
    classifier_input_shape = get_last_layer_shape(variables)
    return jnp.zeros((1, classifier_input_shape[-1]))


def _add_classifier_head(rng, num_classes, variables, ResNet):
    classifier = nn.Dense(num_classes, name="logits")
    classifier_params = classifier.init(rng, _classifier_dummy_input(variables))

    variables = flax.core.unfreeze(variables)
    variables["params"]["logits"] = classifier_params["params"]
    variables = flax.core.freeze(variables)

    # Create the model by merging the backbone and the classifier head
    net = AddLogitsModule(layers=ResNet().layers[:-1], logits=classifier)

    return net, variables


def _normalize_inputs(inputs):
    inputs = inputs - jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
    inputs = inputs / jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))
    return inputs


def resnet(rng, loss_fn, num_classes, model_settings):

    # Initialize the model
    if model_settings.initialization == "imagenet":
        ResNet, variables = jax_resnet.pretrained_resnet(model_settings.num_layers)
    elif model_settings.initialization == "random":
        ResNet = lambda: getattr(
            jax_resnet.resnet, f"ResNet{model_settings.num_layers}"
        )(n_classes=1)
        variables = ResNet().init(rng, jnp.zeros((1, 224, 224, 3)))
    else:
        raise ValueError(
            f"Unknown initialization scheme: {model_settings.initialization}"
        )

    # Replace the last layer with one that has the correct number of classes
    variables = jax_resnet.common.slice_variables(variables, end=-1)
    ResNet, variables = _add_classifier_head(rng, num_classes, variables, ResNet)

    # Partition the parameters into trainable and fixed
    params, fixed_params = hk.data_structures.partition(
        partition_fns[model_settings.trainable_weights], params
    )

    @jax.jit
    def call_resnet(batch: Batch, model_state: ModelState, is_training: bool = True):
        params = hk.data_structures.merge(model_state.params, model_state.fixed_params)
        inputs = _normalize_inputs(batch.inputs)

        logits, variables = ResNet.apply(
            {"params": params, "batch_stats": model_state.state},
            inputs,
            mutable=(["batch_stats"] if is_training else []),
        )

        if is_training:
            model_state = model_state._replace(state=variables["batch_stats"])

        return Predictions(logits), model_state

    model = Model(call_resnet, loss_fn)
    model_state = ModelState(params, fixed_params, variables["batch_stats"], None)

    return model, model_state


if "resnet" not in model_creators:
    model_creators["resnet"] = resnet
