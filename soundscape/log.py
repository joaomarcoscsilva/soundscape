from .composition import Composable
from tqdm import tqdm


def track_over_epoch(keys):
    @Composable
    def _track_over_epoch(values):

        if "epoch_stats" not in values:
            values["epoch_stats"] = {}
            for k in keys:
                values["epoch_stats"][k] = []

        for k in keys:
            values["epoch_stats"][k].append(values[k])

        return values

    return _track_over_epoch


def update_tqdm_bar(keys, length):
    @Composable
    def _update_tqdm_bar(values):
        if "tqdm_bar" not in values:
            values["tqdm_bar"] = tqdm(total=length)

        values["tqdm_bar"].set_postfix({k: values[k] for k in keys})

        return values

    return _update_tqdm_bar
