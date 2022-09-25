import yaml

from . import utils

with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

settings = utils.hash_dict(settings)


class SettingsFunction(object):
    """
    Transform a function of the format f(settings, *args, **kwargs) into a
    function of the format f(*args, **kwargs), with the default settings dict
    partially applied.

    If necessary, f.call can override the default settings dict.
    """

    def __init__(self, fn, settings=settings):
        self.fn = fn
        self.settings = settings

    def __call__(self, *args, **kwargs):
        return self.fn(self.settings, *args, **kwargs)

    def call(self, settings, *args, **kwargs):
        return self.fn(settings, *args, **kwargs)
