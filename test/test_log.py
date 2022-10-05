from soundscape import log, composition

import numpy as np
import pytest


def test_track():
    @composition.Composable
    def f(values):
        a = values["a"]
        b = values["b"]
        c = a + b
        d = a * b
        return {**values, "c": c, "d": d}

    fn = f | log.track(["a", "d"], prefix="pref_")

    values = fn({"a": 1, "b": 2})
    values = fn({**values, "a": 2, "b": 3})
    values = fn({**values, "a": 3, "b": 4})

    assert values == {
        "a": 3,
        "b": 4,
        "c": 7,
        "d": 12,
        "_logs": [
            {"pref_a": 1, "pref_d": 2},
            {"pref_a": 2, "pref_d": 6},
            {"pref_a": 3, "pref_d": 12},
        ],
    }


def test_count_steps():
    @composition.Composable
    def f(values):
        return values

    fn = f | log.count_steps

    values = fn({})
    assert values == {"_step": 0}
    values = fn(values)
    assert values == {"_step": 1}
    values = fn(values)
    assert values == {"_step": 2}


def test_format_digits():
    vals = [1, 12, 2.0, 2.01, 2.000000001, 123.0000001, 123.0123131]
    expected = ["     1", "    12", "2.0000", "2.0100", "2.0000", "123.00", "123.01"]
    formatted = [log.format_digits(np.array([val])[0], 6) for val in vals]
    assert formatted == expected


def test_merge_logs():
    logs = [
        {"a": np.zeros((2, 3)), "b": np.ones((2, 3, 4))},
        {"a": np.zeros((2, 3)), "b": np.ones((2, 3, 4))},
        {"a": np.zeros((2, 3)), "b": np.ones((2, 3, 4))},
        {"c": np.zeros((5, 3))},
        {"c": np.zeros((5, 3))},
    ]

    concat_logs = log.merge_logs(logs, "concat")
    stack_logs = log.merge_logs(logs, "stack")

    assert (concat_logs["a"] == np.zeros((6, 3))).all()
    assert (concat_logs["b"] == np.ones((6, 3, 4))).all()
    assert (concat_logs["c"] == np.zeros((10, 3))).all()
    assert (stack_logs["a"] == np.zeros((3, 2, 3))).all()
    assert (stack_logs["b"] == np.ones((3, 2, 3, 4))).all()
    assert (stack_logs["c"] == np.zeros((2, 5, 3))).all()

    with pytest.raises(ValueError):
        log.merge_logs(logs, "unknown")


def test_mean_keep_dtype():
    x_int32 = np.array([1, 2, 3, 4], dtype=np.int32)
    x_int64 = np.array([1, 2, 3, 4], dtype=np.int64)
    x_float32 = np.array([1, 2, 3, 4], dtype=np.float32)

    assert log.mean_keep_dtype(x_int32) == 2
    assert log.mean_keep_dtype(x_int64) == 2
    assert log.mean_keep_dtype(x_float32) == 2.5

    assert log.mean_keep_dtype(x_int32).dtype == np.int32
    assert log.mean_keep_dtype(x_int64).dtype == np.int64
    assert log.mean_keep_dtype(x_float32).dtype == np.float32


def test_track_progress():
    values = {
        "_step": 8,
        "_logs": [
            {"a": np.array([1]), "b": np.array([2.0]), "c": np.array([3.0])},
            {"a": np.array([1]), "b": np.array([3.0]), "c": np.array([4.0])},
        ],
        "a": 1.0,
    }

    fn = log.track_progress(["a", "c"], total=10)
    values = fn(values)

    assert "_tqdm" in values
    assert values["_tqdm"].total == 10
    assert values["_tqdm"].desc == "a      1 █ c 3.5000: "
