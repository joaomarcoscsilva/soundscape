import inspect
from functools import partial
from typing import Any, Callable, Optional

import jax
import pydantic


class State(dict):
    """
    A dictionary of values that can be used to store the state of a computation.
    """

    def __add__(self, other_dict: dict):
        """
        Combine two states in a new state.
        The original states are not modified.
        """

        return State({**self, **other_dict})

    def select_keys(
        self,
        keys: list,
        key_map: dict | None = None,
    ):
        """
        Select a subset of keys from the state.
        The optional dictionary key_map can be used to rename the keys *before* selecting them.

        Example:
        >>> state = State({"a": 1, "b": 2, "c": 3})
        >>> state.select_keys(["a", "b"])
        {"a": 1, "b": 2}
        >>> state.select_keys(["x", "y"], key_map={"x": "a", "y": "b"})
        {"x": 1, "y": 2}
        """

        if key_map is None:
            return State({k: self[k] for k in keys if k in self})
        else:
            key_map = {**{k: k for k in keys}, **key_map}
            return State({k: self[key_map[k]] for k in keys if key_map[k] in self})

    def __hash__(self):
        """
        Compute the hash of the state.
        """

        return hash(tuple(sorted(self.items())))

    def split(self, first_keys):
        """
        Split the state into two.
        """

        first_dict = {k: v for k, v in self.items() if k in first_keys}
        second_dict = {k: v for k, v in self.items() if k not in first_keys}
        return State(first_dict), State(second_dict)


jax.tree_util.register_pytree_node(
    State,
    lambda state: jax.tree_util.tree_flatten(dict(state)),
    lambda treedef, leaves: State(jax.tree_util.tree_unflatten(treedef, leaves)),
)

StateCallable = Callable[[State], State]
Composable = Any


class Composable:
    """
    A composable function that can be used to build a pipeline.
    All functions involved must have a State -> State signature.

    Each Composable object maintains a list of functions. When called, the Composable object
    applies all functions in the list in order.

    Every function in the list can be marked as traceable or untraceable. An uninterrupted
    sequence of traceable functions can be collapsed into a single function and used in jax
    transformations such as jit or vmap.
    """

    fns: list
    traceable: list

    def __init__(self, fn: StateCallable, traceable: bool = None):
        """
        Create a composable function.
        """

        # Setup the function list
        if isinstance(fn, Composable):
            self.fns = fn.fns
        elif isinstance(fn, list):
            self.fns = fn
        else:
            self.fns = [fn]

        # Setup the traceable flags
        if traceable is None:
            traceable = [True] * len(self.fns)
        elif isinstance(traceable, bool):
            traceable = [traceable] * len(self.fns)

        self.traceable = traceable

    def __call__(self, state: State) -> State:
        """
        Call the composable function.
        """
        for fn in self.fns:
            state = fn(state)
        return state

    def __or__(self, other: Composable) -> Composable:
        """
        Pipeline two composable functions
        """
        if self == identity:
            return other
        elif other == identity:
            return self
        else:
            return Composable(self.fns + other.fns, self.traceable + other.traceable)

    @classmethod
    def untraceable(cls, fn: StateCallable) -> Composable:
        """
        Create an untraceable composable function.
        """

        return cls(fn, traceable=False)

    def _collapse(self) -> StateCallable:
        """
        Collapse all composed functions into a single one.
        """

        def collapsed(state: State) -> State:
            for fn in self.fns:
                state = fn(state)
            return state

        return collapsed

    def _collapse_traceable(self) -> Composable:
        """
        Collapse all consecutive traceable functions into single functions.
        For example, consider the following chain:

        >>> f = traceable_fn1 | traceable_fn2 | untraceable_fn | traceable_fn3

        This function will be collapsed into the following chain:

        >>> f = collapsed_fn1 | untraceable_fn | collapsed_fn2

        Where collapsed_fn1 is the composition of traceable_fn1 and traceable_fn2.
        """

        # List of collapsed Composable functions
        fns = []
        traceables = []

        # The current traceable function being collapsed
        current_fn = identity

        for fn, traceable in zip(self.fns, self.traceable):
            if traceable:
                # If the function is traceable, compose it with the current function
                current_fn = current_fn | Composable(fn)
            else:
                # If the function is not traceable and the current function is not identity,
                # collapse the current function and add it to the list of functions
                if current_fn != identity:
                    fns.append(current_fn._collapse())
                    traceables.append(True)
                    current_fn = identity

                # Add the non-traceable function to the list of functions
                fns.append(fn)
                traceables.append(False)

        # If a traceable function is left, collapse it and add it to the list of functions
        if current_fn != identity:
            fns.append(current_fn._collapse())
            traceables.append(True)

        # Return a new Composable function with all traceable functions collapsed
        return Composable(fns, traceables)

    @staticmethod
    def _jit_fn(
        fn: StateCallable, static_keys: list = [], ignored_keys: list = []
    ) -> StateCallable:
        """
        JIT a single State -> State function.
        """

        # Create a jitted version of the function that takes a state and a static state
        jitted = jax.jit(
            lambda state, static_state: fn(state + static_state),
            static_argnums=1,
        )

        def jitted_statefunction(state: State) -> State:
            """
            Call the jitted composable function.

            The state is split into three parts:
            - ignored state: the state that is not used by the function
            - static state: the state that is used by the function but not modified
            - dynamic state: the state that is used by the function and modified

            All values with keys starting with "_" are automatically placed in the ignored state.

            Any non-traceable values that change frequently (e.g. input filenames) should be
            placed in the ignored state to avoid recompiling the function.
            """

            # Add the keys starting with "_" to the ignored keys
            _ignored_keys = ignored_keys + [k for k in state.keys() if k[0] == "_"]

            # Split the state into ignored, static and dynamic parts
            static_state, callable_state = state.split(static_keys)
            ignored_state, dynamic_state = callable_state.split(_ignored_keys)

            # Call the jitted function with only the dynamic and static parts
            dynamic_state = jitted(dynamic_state, static_state)

            # Return the merged state
            return ignored_state + dynamic_state + static_state

        return jitted_statefunction

    def jit(self, static_keys: list = [], ignored_keys: list = []) -> Composable:
        """
        JIT compile all traceable functions in the composable pipeline.
        Sequential traceable functions are collapsed into a single function before compiling.
        """

        # Collapse all traceable functions
        collapsed = self._collapse_traceable()

        # Create a list of jitted functions
        jitted_fns = collapsed.fns.copy()

        # JIT compile all traceable functions
        for i, fn in enumerate(jitted_fns):
            if collapsed.traceable[i]:
                jitted_fns[i] = Composable._jit_fn(fn, static_keys, ignored_keys)

        # Return a new Composable function with all traceable functions jitted
        return Composable(jitted_fns, collapsed.traceable)

    def vmap(
        self, vmapped_keys: list[str], vmapped_outputs: list[str], in_axes: int = 0
    ) -> Composable:
        """
        Vectorize all functions in the composable pipeline for the specified keys.
        All functions must be traceable.
        """

        if not all(self.traceable):
            raise ValueError("Vectorization is only supported for traceable functions")

        vmapped = jax.vmap(
            lambda vmapped_state, non_vmapped_state: self(
                vmapped_state + non_vmapped_state
            ).split(vmapped_outputs)[0],
            in_axes=(None, 0),
        )

        @Composable
        def vmapped_composable(state: State) -> State:
            vmapped_state, non_vmapped_state = state.split(vmapped_keys)
            outputs = vmapped(non_vmapped_state, vmapped_state)
            return state + outputs

        return vmapped_composable

    def grad(self, input_keys: str | list[str], output_key: str) -> Composable:
        """
        Compute the gradient of the output key with respect to the input key(s).
        """

        def differentiable_fn(differentiable_state, state):
            state = self(state + differentiable_state)
            return state[output_key], state

        grad_fn = jax.value_and_grad(differentiable_fn, has_aux=True)

        @partial(Composable, traceable=all(self.traceable))
        def grad_composable(state: State) -> State:
            differentiable_state, state = state.split(input_keys)
            (output, state), grad = grad_fn(differentiable_state, state)
            return state + State({"grad_" + k: v for k, v in grad.items()})

        return grad_composable


class StateFunction(Composable):
    """
    Wrap an arbitrary function into a State -> State function.
    All named arguments of the function must be present in the state.
    """

    def __init__(
        self,
        fn,
        inputs: dict = {},
        output: str | None = None,
        traceable=True,
        typecheck=True,
    ):
        """
        Create a StateFunction object.

        By default, the input keys are the same as the function arguments, but
        they can be remapped using the input_keys dictionary.

        If fn returns a single value, it is automatically added to the state with the
        same key as the output key. If fn returns a dictionary, the output key can be
        omitted.
        """

        self._fn = fn
        self._inputs = inputs
        self._output = output
        self._function_arguments = inspect.getfullargspec(fn).args

        if typecheck:
            self._wrapped_fn = pydantic.validate_call(
                fn,
                config=pydantic.ConfigDict(arbitrary_types_allowed=True, strict=True),
                validate_return=True,
            )
        else:
            self._wrapped_fn = self._fn

        def call_wrapped(state: State) -> State:
            """
            Wrap the function call, extracting its arguments from the state
            and adding the output to the it.
            """

            # Extract the function arguments from the state
            inputs = state.select_keys(self._function_arguments, key_map=self._inputs)

            # Call the function
            output = self._wrapped_fn(**inputs)

            if self._output is None:
                # If the function returns a dictionary, add it to the state
                return state + State(output)
            else:
                # If the function returns a single value, add it to the state
                # with the specified output key
                return state + State({self._output: output})

        # Initialize the Composable object
        super().__init__(call_wrapped, traceable)

    def inputs(self, input_keys: dict):
        """
        Remap the input keys of the function.
        """

        return StateFunction(self._fn, input_keys, self._output)

    def output(self, output_key: str):
        """
        Remap the output key of the function.
        """

        return StateFunction(self._fn, self._inputs, output_key)

    @classmethod
    def with_output(cls, output: str, *args, **kwargs):
        """
        Convenience decorator to specify the output key of the function.
        """

        def decorator(fn):
            return cls(fn, *args, **kwargs, output=output)

        return decorator


identity = Composable(lambda x: x)
