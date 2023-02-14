import yaml
import inspect
from threading import Lock

# Global settings dictionary
_settings_dict = {}

# Global argument dictionary, used for recursive calls together with a lock
_argdict = {}
_lock = Lock()


def merge_dicts(settings_dict, kwargs):
    """
    Merge two dictionaries, with the second one taking precedence.
    """

    return {k: kwargs[k] if k in kwargs else settings_dict[k] for k in settings_dict}


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

    global _settings_dict

    # Get the function arguments
    argspec = inspect.getfullargspec(fn)

    def wrapped(*args, **kwargs):
        global _argdict, _lock

        # Get the values in the settings dictionary
        settings_args_needed = _settings_dict

        # If there is no **kwargs, select only the arguments that are
        # in the function's signature
        if argspec.varkw is None:
            fn_args = set(argspec.args + argspec.kwonlyargs)
            settings_args_needed = set(_settings_dict).intersection(fn_args)

        # Acquire the lock if it is not already acquired
        must_unlock = False
        if not _lock.locked():
            _lock.acquire()
            must_unlock = True

            # Setup the argument dictionary using the global settings
            # dictionary and the provided keyword arguments
            _argdict = merge_dicts(_settings_dict, kwargs)

        # List the function arguments that were not manually provided
        missing_args = settings_args_needed - set(kwargs.keys())

        # Get the missing values from the argument dictionary
        settings_kwargs = select_keys(missing_args, _argdict)

        # From the arguments that were manually provided, select only the
        # ones that are in the function's signature
        new_kwargs = select_keys(fn_args, kwargs)

        # Call the function
        try:
            results = fn(*args, **new_kwargs, **settings_kwargs)
        except:
            if must_unlock:
                _lock.release()
            raise

        if must_unlock:
            _lock.release()

        return results

    return wrapped


def settings_dict():
    """
    Return the global settings dictionary.
    """

    global _settings_dict
    return _settings_dict


def from_dict(dictionary):
    """
    Set the global settings dictionary.
    """

    global _settings_dict
    _settings_dict = dictionary


def from_file(filename=None):
    """
    Set the global settings dictionary from a YAML file.
    """
    global _settings_dict

    with open(filename, "r") as f:
        _settings_dict = yaml.safe_load(f)
