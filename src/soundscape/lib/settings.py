import yaml

from . import utils

with open("params.yaml", "r") as f:
    settings = yaml.safe_load(f)

settings = utils.hash_dict(settings)
