from typing import Annotated, Generic, TypeVar, TypeVarTuple

import chex
import jax
import pydantic


# Monkey patch weakref support into the validate_call function
class NewValidateCallWrapper(pydantic._internal._validate_call.ValidateCallWrapper):
    __slots__ = ("__weakref__",)


pydantic._internal._validate_call.ValidateCallWrapper = NewValidateCallWrapper


def validate_array_tree(x: chex.ArrayTree) -> chex.ArrayTree:
    leaves, treedef = jax.tree_util.tree_flatten(x)
    for leaf in leaves:
        if not isinstance(leaf, chex.Array):
            raise ValueError(f"Expected chex.Array, got {type(leaf)}")
    return x


ArrayTree = Annotated[
    chex.ArrayTree,
    pydantic.PlainValidator(validate_array_tree),
]

Array = chex.Array

validate_call = pydantic.validate_call(
    config=pydantic.ConfigDict(arbitrary_types_allowed=True, strict=True),
    validate_return=True,
)
