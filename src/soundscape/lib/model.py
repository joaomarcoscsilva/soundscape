import haiku as hk
from jax import numpy as jnp
import jax
from oryx.core import sow

from .settings import SettingsFunction
from . import sow_transforms


def partition(apply_fn, partition_fn, params):

    params, fixed_params = hk.data_structures.partition(partition_fn, params)

    def _apply_fn(params, fixed_params, *args, **kwargs):
        params = hk.data_structures.merge(params, fixed_params)
        return apply_fn(params, *args, **kwargs)

    return _apply_fn, params, fixed_params


partition_fns = {
    "all": lambda m, n, p: True,
    "none": lambda m, n, p: False,
    "head": lambda m, n, p: "logits" in m,
}


@SettingsFunction
def resnet(settings, rng, *args, **kwargs):
    def _resnet(x, is_training=True):
        version = int(settings["model"]["name"].split("_")[1])
        net = hk.nets.ResNet(
            *args,
            num_classes=settings["data"]["num_classes"],
            **{**kwargs, **hk.nets.ResNet.CONFIGS[version]}
        )
        return net(x, is_training=is_training)

    init_fn, apply_fn = hk.transform_with_state(_resnet)

    input_shape = (224, 224, 3) if settings["data"]["downsample"] else (860, 256, 3)
    params, state = init_fn(rng, jnp.zeros((1, *input_shape)), True)

    partition_fn = partition_fns[settings["train"]["partition_fn"]]
    apply_fn, params, fixed_params = partition(apply_fn, partition_fn, params)

    apply_fn = sow_transforms.tuple_to_sow(
        apply_fn, names=["logits", "state"], tag="output"
    )

    return (params, fixed_params, state), apply_fn
