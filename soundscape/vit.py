from transformers import FlaxViTForImageClassification
import haiku as hk

from .composition import Composable
from .settings import from_file

settings_fn, settings_dict = from_file()

# Partitioning functions to define which parameters are trainable
partition_fns = {
    "all": lambda m, n, p: True,
    "none": lambda m, n, p: False,
    "head": lambda m, n, p: "logits" in m,
}


@settings_fn
def vit(rng, *, initialization, num_classes, trainable_weights):

    vit = FlaxViTForImageClassification.from_pretrained(
        initialization, num_labels=num_classes
    )

    params = vit.params

    params, fixed_params = hk.data_structures.partition(
        partition_fns[trainable_weights], params
    )

    @Composable
    def call_vit(values):
        params = values["params"]
        fixed_params = values["fixed_params"]

        params = hk.data_structures.merge(params, fixed_params)

        inputs = values["inputs"]
        inputs = inputs.transpose((0, 3, 1, 2))
        inputs = (inputs - 0.5) / 0.5
        preds = vit(inputs, params=params)

        return {**values, **preds}

    return call_vit, {"params": params, "fixed_params": fixed_params}
