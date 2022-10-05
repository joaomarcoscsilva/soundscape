from transformers import FlaxViTForImageClassification
import haiku as hk
import pickle
import jax
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


@settings_fn
def vit(rng, *, initialization, num_classes, trainable_weights, convert_tf_params):
    """
    Return a ViT composable function that produces logits and batch stats.
    Also return a dictionary with the initial variables.

    Parameters:
    ----------
    rng: jax.random.PRNGKey
        The random number generator key.

    Settings:
    --------
    initialization: str
        The initialization scheme to use.
    num_classes: int
        The number of classes to predict.
    trainable_weights: str
        The weights to train. Can be "all", "none", or "head"
    convert_tf_params: bool
    """

    # Load the ViT model.
    vit = FlaxViTForImageClassification.from_pretrained(
        initialization, num_labels=num_classes
    )
    params = vit.params

    if convert_tf_params:
        with open("vit_b16.pkl", "rb") as f:
            tf_params = pickle.load(f)

        if num_classes != 12:
            params["vit"] = jax.tree_util.tree_map(
                lambda x: jnp.array(x), tf_params["vit"]
            )
        else:
            params = jax.tree_util.tree_map(lambda x: jnp.array(x), tf_params)

    # Partition the parameters into trainable and fixed
    params, fixed_params = hk.data_structures.partition(
        partition_fns[trainable_weights], params
    )

    @Composable
    def call_vit(values):
        """
        Resnet call function.

        Parameters:
        ----------
        values["params"]: dict
            The trainable parameters of the model.
        values["fixed_params"]: dict
            The fixed parameters of the model.
        values["inputs"]: jnp.ndarray
            The inputs to the model.
        """

        inputs = values["inputs"]
        params = values["params"]
        fixed_params = values["fixed_params"]

        # Merge the trainable and fixed parameters
        params = hk.data_structures.merge(params, fixed_params)

        # Change the input format to channels first
        inputs = inputs.transpose((0, 3, 1, 2))

        # Normalize the inputs
        # inputs = (inputs - 0.5) / 0.5

        # Apply the model
        preds = vit(inputs, params=params)

        return {**values, **preds}

    return call_vit, {"params": params, "fixed_params": fixed_params}
