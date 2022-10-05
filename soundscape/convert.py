import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


param_shapes = {
    "classifier": {"bias": (12,), "kernel": (768, 12)},
    "vit": {
        "embeddings": {
            "cls_token": (1, 1, 768),
            "patch_embeddings": {
                "projection": {"bias": (768,), "kernel": (16, 16, 3, 768)}
            },
            "position_embeddings": (1, 197, 768),
        },
        "encoder": {
            "layer": {
                "0": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "1": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "10": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "11": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "2": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "3": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "4": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "5": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "6": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "7": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "8": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
                "9": {
                    "attention": {
                        "attention": {
                            "key": {"bias": (768,), "kernel": (768, 768)},
                            "query": {"bias": (768,), "kernel": (768, 768)},
                            "value": {"bias": (768,), "kernel": (768, 768)},
                        },
                        "output": {"dense": {"bias": (768,), "kernel": (768, 768)}},
                    },
                    "intermediate": {"dense": {"bias": (3072,), "kernel": (768, 3072)}},
                    "layernorm_after": {"bias": (768,), "scale": (768,)},
                    "layernorm_before": {"bias": (768,), "scale": (768,)},
                    "output": {"dense": {"bias": (768,), "kernel": (3072, 768)}},
                },
            }
        },
        "layernorm": {"bias": (768,), "scale": (768,)},
    },
}

import copy

params = copy.deepcopy(param_shapes)


def dense(name):
    return {"kernel": f"{name}/kernel:0", "bias": f"{name}/bias:0"}


def layernorm(name):
    return {"scale": f"{name}/gamma:0", "bias": f"{name}/beta:0"}


params["classifier"] = dense("head")


params["vit"]["embeddings"]["cls_token"] = "class_token/cls:0"
params["vit"]["embeddings"]["patch_embeddings"]["projection"] = dense("embedding")
params["vit"]["embeddings"][
    "position_embeddings"
] = "Transformer/posembed_input/pos_embedding:0"

params["vit"]["layernorm"] = layernorm("Transformer/encoder_norm")

for layer_num in params["vit"]["encoder"]["layer"]:
    layer = params["vit"]["encoder"]["layer"][layer_num]

    layer["attention"]["attention"]["key"] = dense(
        f"Transformer/encoderblock_{layer_num}/MultiHeadDotProductAttention_1/key"
    )
    layer["attention"]["attention"]["query"] = dense(
        f"Transformer/encoderblock_{layer_num}/MultiHeadDotProductAttention_1/query"
    )
    layer["attention"]["attention"]["value"] = dense(
        f"Transformer/encoderblock_{layer_num}/MultiHeadDotProductAttention_1/value"
    )
    layer["attention"]["output"]["dense"] = dense(
        f"Transformer/encoderblock_{layer_num}/MultiHeadDotProductAttention_1/out"
    )

    layer["intermediate"]["dense"] = dense(
        f"Transformer/encoderblock_{layer_num}/Dense_0"
    )
    layer["output"]["dense"] = dense(f"Transformer/encoderblock_{layer_num}/Dense_1")
    layer["layernorm_before"] = layernorm(
        f"Transformer/encoderblock_{layer_num}/LayerNorm_0"
    )
    layer["layernorm_after"] = layernorm(
        f"Transformer/encoderblock_{layer_num}/LayerNorm_2"
    )

from vit_keras import vit
import jax

model = vit.vit_b16(
    image_size=224,
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=12,
    activation="softmax",
)

vars = {var.name: var for var in model.variables}
params = jax.tree_util.tree_map(lambda name: vars[name], params)
import numpy as np
params = jax.tree_util.tree_map(lambda x: np.array(x), params)


import pickle

pickle.dump(params, open("vit_b16.pkl", "wb"))
