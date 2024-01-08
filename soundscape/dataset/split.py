import jax
import numpy as np
import pandas as pd
from jax import random


def genereate_splits(
    num_samples: int, rng: jax.Array, splits: list[str], fractions: list[float]
):
    """
    Split a class into train, validation and test sets.
    Returns an array of strings with the split name for each sample.
    """

    samples = np.floor(np.array(fractions) * num_samples).astype(int)

    # Correct for rounding errors
    samples[0] += num_samples - np.sum(samples)

    splits = np.concatenate(
        [np.repeat(split, samples[i]) for i, split in enumerate(splits)]
    )

    splits = splits[random.permutation(rng, len(splits))]

    return splits


def split_dataframe(
    labels_df: pd.DataFrame,
    split_seed: int,
    splits: list[str],
    fractions: list[float],
    stratify: bool,
) -> pd.DataFrame:
    """
    Split a class into train, validation and test sets
    """

    rng = random.PRNGKey(split_seed)

    if stratify:
        df_classes = labels_df.groupby("class")

        for class_id, df_class in df_classes:
            rng, split_rng = random.split(rng)
            sample_splits = genereate_splits(
                len(df_class), split_rng, splits, fractions
            )
            labels_df.loc[df_class.index, "split"] = sample_splits

    else:
        df_files = list(labels_df.groupby("file"))
        splits = genereate_splits(len(df_files), rng, splits, fractions)
        for i, (file_name, df_file) in enumerate(df_files):
            labels_df.loc[df_file.index, "split"] = splits[i]

    return labels_df
