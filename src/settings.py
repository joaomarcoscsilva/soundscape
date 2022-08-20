import yaml

import utils

with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

settings = utils.hash_dict(settings)
