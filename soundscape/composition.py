from typing import Protocol, Any, Union, Optional
import jax

Value = Any
Values = Any


class Composable:
    """
    A composable function that can be used to build a pipeline.
    The function will always receive and return a dictionary of values.

    Composable functions can be pipelined using the | operator.
    """

    def __init__(self, fn, key: Optional[str] = None):
        """
        Create a composable function.

        Parameters:
        ----------
        fn: function or Composable
            The function to be wrapped.
            If a Composable is provided, nothing will be wrapped.

        key: str, optional
            If provided, the function's output will be stored in
            the dictionary under the provided key.

            If not provided, the function's output will be returned
            as is.
        """

        if isinstance(fn, Composable):
            self.fn = fn.fn
        elif key is None:
            self.fn = fn
        else:
            self.fn = lambda values: {**values, key: fn(values)}

    def __call__(self, values: Values, **kwargs) -> Values:
        return self.fn(values, **kwargs)

    def __or__(self, other):
        """
        Pipeline two composable functions
        """
        return Composable(lambda values: other(self(values)))


class hashable_dict(dict):
    """
    A hashable dictionary.
    """

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def split_dict(values, keys):
    """
    Split a dictionary into two dictionaries.

    Parameters:
    ----------
    keys: list
        The keys to be included in the second dictionary.
    """

    d1 = {k: v for k, v in values.items() if k not in keys}
    d2 = {k: v for k, v in values.items() if k in keys}
    return d1, d2


def jit(function, static_keys=[], ignored_keys=[]):
    """
    JIT compile a composable function.

    Parameters:
    ----------
    function: Composable
        The function to be JIT compiled.
    static_keys: list
        The keys that will be passed as static arguments to the JIT compiled function.
    ignored_keys: list
        The keys whose values will not be passed to the compiled function.
        Any key that stats with an underscore will be ignored by default.

    Returns:
    -------
    Composable
        The JIT compiled function.
    """

    # Define a helper function that can be compiled directly by jax.jit
    def jittable_function(values, static_values):
        return function({**values, **static_values})

    # JIT compile the helper function
    jitted_function = jax.jit(jittable_function, static_argnums=1)

    @Composable
    def _function(values):
        """
        The JIT compiled function.
        """

        # Add the keys that start with an underscore to the list of ignored keys
        _ignored_keys = ignored_keys + [k for k in values.keys() if k[0] == "_"]

        # Split the dictionary into static, ignored and dynamic values
        values, static_values = split_dict(values, static_keys)
        values, ignored_values = split_dict(values, _ignored_keys)

        # Call the JIT compiled function
        values = jitted_function(values, hashable_dict(static_values))

        # Return the combined values
        return {**values, **ignored_values, **static_values}

    return _function


def grad(function, input_key: str, output_key: str):
    """
    Compute the gradient of a composable function.

    The gradient will be stored in the dictionary under the key "grad".

    Parameters:
    ----------
    function: Composable
        The function to be differentiated.
    input_key: str
        The key containing the parameters with respect
        to which the function will be differentiated.
    output_key: str
        The key containing the value that will be differentiated.

    Returns:
    -------
    Composable
        The function that computes the gradient.
    """

    # Define a helper function that can be differentiated directly by jax.grad
    def _differentiable_function(differentiable_values, values):
        output = function({**values, input_key: differentiable_values})
        return output[output_key], {**values, **output}

    # Differentiate the helper function
    grad_fn = jax.value_and_grad(_differentiable_function, has_aux=True)

    @Composable
    def _grad_function(values):
        """
        The function that computes the gradient.
        """

        # Compute the gradient and the new values
        (_, new_values), grad = grad_fn(values[input_key], values)

        # Return the new values and the gradient
        return {**new_values, "grad": grad}

    return _grad_function


@Composable
def identity(values):
    """
    A composable function that does nothing.
    """
    return values
