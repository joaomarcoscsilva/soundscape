import pickle

import haiku as hk
import jax
from jax import numpy as jnp
from transformers import FlaxViTForImageClassification

from ..types import Batch
from .base_model import Model, ModelState, Predictions, model_creators, partition_fns


def vit(rng, loss_fn, num_classes, model_settings):
    # Load the ViT model.
    ViT = FlaxViTForImageClassification.from_pretrained(
        model_settings.initialization, num_labels=num_classes
    )
    params = ViT.params

    if model_settings.convert_tf_params:
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
        partition_fns[model_settings.trainable_weights], params
    )

    def call_vit(batch: Batch, model_state: ModelState, is_training: bool = True):
        params = hk.data_structures.merge(model_state.params, model_state.fixed_params)
        inputs = batch.inputs.transpose((0, 3, 1, 2))
        logits = ViT(inputs, params=params)
        return Predictions(logits=logits), model_state

    model = Model(call_vit, loss_fn)
    model_state = ModelState(params, fixed_params, None, None)

    return model, model_state


model_creators["vit"] = vit
