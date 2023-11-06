import pydantic


# Monkey patch weakref support into the validate_call function
class NewValidateCallWrapper(pydantic._internal._validate_call.ValidateCallWrapper):
    __slots__ = ("__weakref__",)


pydantic._internal._validate_call.ValidateCallWrapper = NewValidateCallWrapper
