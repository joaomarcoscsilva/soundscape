from jax import numpy as jnp
import jax
import pytest

from soundscape.lib import train_loop


@pytest.mark.parametrize(
    "state,new_aux,expected",
    [
        (None, {"a": 1.0, "b": 2.0}, {"a": [1.0], "b": [2.0]}),
        (
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            {"a": 3.0, "b": 5.0},
            {"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]},
        ),
    ],
)
def test_append_fn(state, new_aux, expected):
    assert train_loop.append_fn(state, new_aux) == expected
