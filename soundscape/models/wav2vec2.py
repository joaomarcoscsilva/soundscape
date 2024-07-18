import flax
import haiku as hk
import jax
import librosa
import transformers
from flax import linen as nn
from jax import numpy as jnp

from ..types import Batch
from .base_model import Model, ModelState, Predictions, model_creators, partition_fns


class AddLogitsModule(nn.Module):
    """
    Add a linear layer to the end of a model.
    """

    base_model: callable
    logits: callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = self.base_model(x).last_hidden_state.mean(-2)
        return self.logits(x)


def _add_classifier_head(rng, num_classes, params, base_model):
    classifier = nn.Dense(num_classes, name="logits")
    net = AddLogitsModule(base_model, logits=classifier)
    net_params = net.init(rng, jnp.zeros((1, 16000)))

    for key in params:
        net_params["params"][key] = params[key]

    return net, net_params


def wav2vec2(rng, loss_fn, num_classes, model_settings):
    config = transformers.Wav2Vec2Config.from_pretrained(model_settings.initialization)
    config.do_stable_layer_norm = True
    config.feat_extract_norm = "layer"
    base_model = transformers.FlaxWav2Vec2Model.from_pretrained(
        model_settings.initialization, config=config
    )
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), base_model.params)
    wav2vec2, params = _add_classifier_head(rng, num_classes, params, base_model)

    # Partition the parameters into trainable and fixed
    params, fixed_params = hk.data_structures.partition(
        partition_fns[model_settings.trainable_weights], params
    )

    def call_model(batch: Batch, model_state: ModelState, is_training: bool = True):
        params = hk.data_structures.merge(model_state.params, model_state.fixed_params)
        audio = batch["inputs"][..., 0]
        logits = wav2vec2.apply(params, audio)
        return Predictions(logits=logits), model_state

    model = Model(call_model, loss_fn)
    model_state = ModelState(params, fixed_params, None, None)

    return model, model_state


model_creators["wav2vec2"] = wav2vec2
