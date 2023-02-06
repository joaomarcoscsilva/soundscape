import yaml
import inspect
import sys
from threading import Lock

settings_dict = {}

argdict = {}
lock = Lock()


def merge_dicts(settings_dict, kwargs):
    return {k: kwargs[k] if k in kwargs else settings_dict[k] for k in settings_dict}


def select_keys(keys, dictionary):
    return {k: v for k, v in dictionary.items() if k in keys}


def settings_fn(fn):
    global settings_dict
    argspec = inspect.getfullargspec(fn)

    def wrapped(*args, **kwargs):
        global argdict, lock

        settings_args_needed = settings_dict
        if argspec.varkw is None:
            fn_args = set(argspec.args + argspec.kwonlyargs)
            settings_args_needed = set(settings_dict).intersection(fn_args)

        must_unlock = False

        if not lock.locked():
            lock.acquire()
            argdict = merge_dicts(settings_dict, kwargs)
            must_unlock = True

        missing_args = settings_args_needed - set(kwargs.keys())
        settings_kwargs = select_keys(missing_args, argdict)
        new_kwargs = select_keys(fn_args, kwargs)

        try:
            results = fn(*args, **new_kwargs, **settings_kwargs)
        except:
            if must_unlock:
                lock.release()
            raise

        if must_unlock:
            lock.release()

        return results

    return wrapped


def from_dict(dictionary):
    global settings_dict
    settings_dict = dictionary


def from_file(filename=None):
    global settings_dict

    if filename is None and sys.argv[-1].endswith(".yaml"):
        filename = sys.argv[1]

    if filename is not None:
        with open(filename, "r") as f:
            settings_dict = yaml.safe_load(f)
