import yaml
import inspect

argdict = {}


def merge_dicts(settings_dict, kwargs):
    return {k: kwargs[k] if k in kwargs else settings_dict[k] for k in settings_dict}


def select_keys(keys, dictionary):
    return {k: v for k, v in dictionary.items() if k in keys}


def from_dict(settings_dict):
    def transform(fn):
        argspec = inspect.getfullargspec(fn)

        settings_args_needed = settings_dict

        if argspec.varkw is None:
            fn_args = set(argspec.args + argspec.kwonlyargs)
            settings_args_needed = set(settings_dict).intersection(fn_args)

        def wrapped(*args, **kwargs):
            global argdict
            clean = False

            if argdict == {}:
                argdict = merge_dicts(settings_dict, kwargs)
                clean = True
                print(argdict)

            missing_args = settings_args_needed - set(kwargs.keys())
            settings_kwargs = select_keys(missing_args, argdict)
            kwargs = select_keys(fn_args, kwargs)

            try:
                results = fn(*args, **kwargs, **settings_kwargs)
            except:
                if clean:
                    argdict = {}
                raise

            if clean:
                argdict = {}

            return results

        return wrapped

    return transform, settings_dict


def from_file(filename):
    with open(filename, "r") as f:
        settings_dict = yaml.safe_load(f)

    return from_dict(settings_dict)
