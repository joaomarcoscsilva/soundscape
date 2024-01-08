import jax
import numpy as np
import pandas as pd

from soundscape.dataset import split

KEY = jax.random.PRNGKey(0)


def test_generate_splits():
    splits = split.genereate_splits(
        1000, KEY, ["train", "val", "test"], [0.8, 0.1, 0.1]
    )

    assert len(splits) == 1000
    assert (splits == "train").sum() == 800
    assert (splits == "val").sum() == 100
    assert (splits == "test").sum() == 100


def test_split_df_stratified():
    for i in range(20):
        df = pd.DataFrame({"class": ["a"] * 400 + ["b"] * 600})
        df = split.split_dataframe(
            df, i, ["train", "val", "test"], [0.8, 0.1, 0.1], stratify=True
        )

        assert len(df) == 1000
        assert (df["split"] == "train").sum() == 800
        assert (df["split"] == "val").sum() == 100
        assert (df["split"] == "test").sum() == 100

        assert (df[df["class"] == "a"]["split"] == "train").sum() == int(400 * 0.8)
        assert (df[df["class"] == "a"]["split"] == "val").sum() == int(400 * 0.1)
        assert (df[df["class"] == "a"]["split"] == "test").sum() == int(400 * 0.1)

        assert (df[df["class"] == "b"]["split"] == "train").sum() == int(600 * 0.8)
        assert (df[df["class"] == "b"]["split"] == "val").sum() == int(600 * 0.1)
        assert (df[df["class"] == "b"]["split"] == "test").sum() == int(600 * 0.1)


def test_split_df_not_stratified():
    all_checks = []

    for i in range(20):
        df = pd.DataFrame(
            {"class": ["a"] * 400 + ["b"] * 600, "file": list(range(500)) * 2}
        )
        df = split.split_dataframe(
            df, i, ["train", "val", "test"], [0.8, 0.1, 0.1], stratify=False
        )

        assert len(df) == 1000
        assert (df["split"] == "train").sum() == 800
        assert (df["split"] == "val").sum() == 100
        assert (df["split"] == "test").sum() == 100

        all_checks.append(
            (df[df["class"] == "a"]["split"] == "train").sum() == int(400 * 0.8)
        )
        all_checks.append(
            (df[df["class"] == "a"]["split"] == "val").sum() == int(400 * 0.1)
        )
        all_checks.append(
            (df[df["class"] == "a"]["split"] == "test").sum() == int(400 * 0.1)
        )

        all_checks.append(
            (df[df["class"] == "b"]["split"] == "train").sum() == int(600 * 0.8)
        )
        all_checks.append(
            (df[df["class"] == "b"]["split"] == "val").sum() == int(600 * 0.1)
        )
        all_checks.append(
            (df[df["class"] == "b"]["split"] == "test").sum() == int(600 * 0.1)
        )

        assert df.groupby("file")["split"].nunique().max() == 1

    assert not all(all_checks)
