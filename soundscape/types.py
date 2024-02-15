from typing import Any, NamedTuple, Optional, TypedDict

import jax
from numpy.typing import NDArray

PyTree = Any


class Batch(TypedDict, total=False):
    inputs: jax.Array
    labels: jax.Array
    label_probs: jax.Array
    rng: jax.Array
    rngs: jax.Array

    _files: NDArray
    _ids: NDArray


class ModelState(NamedTuple):
    params: PyTree
    fixed_params: PyTree

    state: Optional[PyTree]
    optim_state: Optional[PyTree]


class Predictions(TypedDict, total=False):
    logits: PyTree
    ...
