import yaml
import inspect
import argparse
from threading import Lock
from contextlib import nullcontext


_global_settings_stack = []


def settings_dict():
    global _global_settings_stack
    return _global_settings_stack[-1] if _global_settings_stack else {}


class Settings:
    def __init__(self, settings_dict):
        self.settings_dict = settings_dict

    def __enter__(self):
        global _global_settings_stack
        head = settings_dict()
        _global_settings_stack.append({**head, **self.settings_dict})

    def __exit__(self, exc_type, exc_value, traceback):
        global _global_settings_stack
        _global_settings_stack.pop()

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            settings_dict = yaml.safe_load(f)
        return cls(settings_dict)

    @classmethod
    def from_command_line(cls):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "settings", type=str, help="YAML settings file", default=None
        )
        parser.add_argument(
            "-k",
            "--key",
            type=str,
            help="Key to set",
            action="append",
            nargs=2,
            default=[],
        )
        args = parser.parse_args()

        settings_dict = {}

        if args.settings is not None:
            with open(args.settings, "r") as f:
                settings_dict = yaml.safe_load(f)

        settings_dict = {**settings_dict, **{k: yaml.safe_load(v) for k, v in args.key}}

        return cls(settings_dict)


def select_keys(keys, dictionary):
    """
    Select a subset of keys from a dictionary.
    """

    return {k: v for k, v in dictionary.items() if k in keys}


def settings_fn(fn):
    """
    Decorator for functions that use settings.

    Any keyword-only argument of the function will be taken from the global
    settings dictionary, if it is not manually provided.

    If a value is provided manually, it will be used instead of the global
    dictionary, including for recursive calls.
    """

    # Get the function arguments
    argspec = inspect.getfullargspec(fn)
    fn_args = set(argspec.kwonlyargs)

    def wrapped(*args, **kwargs):
        global _global_settings_stack

        head = settings_dict()

        if not all([((k in head) and (kwargs[k] == head[k])) for k in kwargs]):
            new_head = {**head, **kwargs}
            settings_context = Settings(new_head)
        else:
            settings_context = nullcontext()

        with settings_context:
            missing_args = set(fn_args) - set(kwargs.keys())
            settings_kwargs = select_keys(missing_args, head)
            new_kwargs = select_keys(fn_args, kwargs)
            results = fn(*args, **new_kwargs, **settings_kwargs)

        return results

    return wrapped