from typing import Protocol, Any, Union, Optional
import jax

Value = Any
Values = Any


class SimpleFunction(Protocol):
    def __call__(self, values: Values) -> Value:
        ...


class ComposableFunction(Protocol):
    def __call__(self, values: Values) -> Values:
        ...


class Composable:
    def __init__(
        self, fn: Union[SimpleFunction, ComposableFunction], key: Optional[str] = None
    ):
        if isinstance(fn, Composable):
            self.fn = fn.fn
        elif key is None:
            self.fn = fn
        else:
            self.fn = lambda values: {**values, key: fn(values)}

    def __call__(self, values: Values) -> Values:
        return self.fn(values)

    def __or__(self, other: ComposableFunction):
        return Composable(lambda values: other(self(values)))


class hashable_dict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def split_dict(values, keys):
    d1 = {k: v for k, v in values.items() if k not in keys}
    d2 = {k: v for k, v in values.items() if k in keys}
    return d1, d2


def jit(
    function: ComposableFunction, static_keys=[], ignored_keys=[]
) -> ComposableFunction:
    def jittable_function(values, static_values):
        return function({**values, **static_values})

    jitted_function = jax.jit(jittable_function, static_argnums=1)

    @Composable
    def _function(values):
        _ignored_keys = ignored_keys + [k for k in values.keys() if k[0] == "_"]

        values, static_values = split_dict(values, static_keys)
        values, ignored_values = split_dict(values, _ignored_keys)

        values = jitted_function(values, hashable_dict(static_values))
        return {**values, **ignored_values, **static_values}

    return _function


def grad(
    function: ComposableFunction, input_key: str, output_key: str
) -> ComposableFunction:
    def _differentiable_function(differentiable_values, values):
        output = function({**values, input_key: differentiable_values})
        return output[output_key], {**values, **output}

    @Composable
    def _grad_function(values):
        grad_fn = jax.value_and_grad(_differentiable_function, has_aux=True)
        (_, new_values), grad = grad_fn(values[input_key], values)
        return {**new_values, "grad": grad}

    return _grad_function


@Composable
def identity(values):
    return values
