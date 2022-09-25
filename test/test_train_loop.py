from jax import numpy as jnp
import jax

from soundscape.lib import train_loop


def test_accumulate_state():

    state = {"a": jnp.array([[0.0, 0.0], [0.0, 0.0]]), "b": jnp.array([[0.0], [0.0]])}

    new_state = train_loop.accumulate_state(None, state)
    assert new_state == state

    aux = {"a": jnp.array([[1.0, 1.0], [1.0, 1.0]]), "b": jnp.array([[1.0], [1.0]])}

    new_state = train_loop.accumulate_state(state, aux)

    assert jnp.all(
        new_state["a"] == jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    )
    assert jnp.all(new_state["b"] == jnp.array([[0.0], [0.0], [1.0], [1.0]]))
