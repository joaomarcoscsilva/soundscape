import jax
import numpy as np

from soundscape import log


def add_sample_values(logger):
    return logger


def assert_tree_equal(a, b):
    flat_a, treedef_a = jax.tree_util.tree_flatten(a)
    flat_b, treedef_b = jax.tree_util.tree_flatten(b)

    assert treedef_a == treedef_b

    for i in range(len(flat_a)):
        assert np.all(flat_a[i] == flat_b[i])


def test_simple_logger():
    logger = log.Logger()
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2])})
    logger.update({"a": np.array([3]), "b": np.array([4])})
    logger.update({"a": np.array([5]), "b": np.array([6])})
    results = logger.close()

    assert_tree_equal(
        results,
        {
            "a": np.array([1, 3, 5]),
            "b": np.array([2, 4, 6]),
        },
    )

    # Assert that the logger can be reset
    logger.restart()
    assert logger._logs == {}
    assert logger.close() == {}


def test_incremental_additions():
    logger = log.Logger()
    logger.restart()

    logger.update({"a": np.array([1])})
    assert_tree_equal(logger.merge(), {"a": np.array([1])})

    logger.update({"a": np.array([2]), "b": np.array([3])})
    assert_tree_equal(logger.merge(), {"a": np.array([1, 2]), "b": np.array([3])})

    logger.update({"b": np.array([5])})
    assert_tree_equal(logger.merge(), {"a": np.array([1, 2]), "b": np.array([3, 5])})


def test_logged_keys():
    logger = log.Logger(logged_keys=["a"])
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2])})
    logger.update({"a": np.array([3]), "b": np.array([4])})
    logger.update({"a": np.array([5]), "b": np.array([6])})
    results = logger.close()

    assert_tree_equal(
        results,
        {
            "a": np.array([1, 3, 5]),
        },
    )


def test_serialize():
    logger = log.TrainingLogger(saved_keys=["a", "b"])
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2]), "c": np.array([3])})
    logger.update({"a": np.array([3]), "b": np.array([4]), "c": np.array([5])})
    logger.update({"a": np.array([5]), "b": np.array([6]), "c": np.array([7])})
    logger.update({"a": np.array([1]), "c": np.array([3])}, prefix="val_")
    serialized = logger.serialized()

    assert serialized == '{"a": [1, 3, 5], "b": [2, 4, 6], "val_a": [1]}'


def test_pbar():
    logger = log.Logger(pbar_len=4, pbar_keys=["a"])
    logger.restart()

    logger.update({"b": np.array([2])})
    logger.update({"a": np.array([1.0])})
    logger.update({"a": np.array([3.0]), "b": np.array([4])})

    assert logger.pbar.total == 4
    assert logger.pbar.n == 3
    assert logger.pbar.desc == "a 2.0000: "

    logger.update({"a": np.array([5.0]), "b": np.array([6])})

    assert logger.pbar.n == 4
    assert logger.pbar.desc == "a 3.0000: "


def test_stack_logger():
    logger = log.Logger(merge_fn="stack")
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2])})
    logger.update({"a": np.array([3]), "b": np.array([4])})
    logger.update({"a": np.array([5]), "b": np.array([6])})
    results = logger.close()

    assert_tree_equal(
        results,
        {
            "a": np.array([[1], [3], [5]]),
            "b": np.array([[2], [4], [6]]),
        },
    )


def test_prefix():
    logger = log.Logger()
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2])})
    logger.update({"a": np.array([3]), "b": np.array([4])})
    logger.update({"a": np.array([5]), "b": np.array([6])}, prefix="test_")
    results = logger.merge()

    assert_tree_equal(
        results,
        {
            "a": np.array([1, 3]),
            "b": np.array([2, 4]),
            "test_a": np.array([5]),
            "test_b": np.array([6]),
        },
    )


def test_format_digits():
    vals = [
        1,
        12,
        2.0,
        2.01,
        2.000000001,
        123.0000001,
        123.0123131,
        np.float32("nan"),
        np.float32("inf"),
        np.float32("-inf"),
    ]
    expected = [
        "     1",
        "    12",
        "2.0000",
        "2.0100",
        "2.0000",
        "123.00",
        "123.01",
        "   nan",
        "   inf",
        "  -inf",
    ]
    formatted = [log.format_digits(np.array([val])[0], 6) for val in vals]
    assert formatted == expected


def test_scalar():
    logger = log.Logger()
    logger.restart()

    logger.update({"a": 1, "b": np.array([2])})
    logger.update({"a": 3, "b": np.array([4])})
    logger.update({"a": 5, "b": np.array([6])})
    results = logger.close()

    assert_tree_equal(
        results,
        {
            "a": np.array([1, 3, 5]),
            "b": np.array([2, 4, 6]),
        },
    )


def test_nan_early_stopping():
    logger = log.TrainingLogger(nan_metrics=["a"])
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2])})
    assert not logger.early_stop()

    logger.update({"a": np.array([1]), "b": np.array([np.nan])})
    assert not logger.early_stop()

    logger.update({"a": np.array([np.nan]), "b": np.array([np.nan])})
    assert logger.early_stop()

    logger.restart()
    assert not logger.early_stop()

    logger.update({"a": np.array([np.nan]), "b": np.array([1])}, prefix="val_")
    assert logger.early_stop()


def test_patience_early_stopping():
    logger = log.TrainingLogger(
        optimizing_metric="a", patience=2, optimizing_mode="min"
    )
    logger.restart()

    logger.update({"a": np.array([3])})
    assert not logger.early_stop()

    logger.update({"a": np.array([2])})
    assert not logger.early_stop()

    logger.update({"a": np.array([1])})
    assert not logger.early_stop()

    logger.update({"a": np.array([1])})
    assert not logger.early_stop()

    logger.update({"a": np.array([1])})
    assert logger.early_stop()


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


def test_optimizing_metric():
    logger = log.TrainingLogger(optimizing_metric="a", optimizing_mode="max")
    logger.restart()

    logger.update({"a": np.array([[1]])})
    assert logger.improved()
    assert logger.best() == logger.latest() == 1
    assert logger.best_epoch() == 0

    logger.update({"a": np.array([[2]])})
    assert logger.improved()
    assert logger.best() == logger.latest() == 2
    assert logger.best_epoch() == 1

    logger.update({"a": np.array([[3]])})
    assert logger.improved()
    assert logger.best() == logger.latest() == 3
    assert logger.best_epoch() == 2

    logger.update({"a": np.array([[3]])})
    assert logger.improved()
    assert logger.best() == logger.latest() == 3
    assert logger.best_epoch() == 2

    logger.update({"a": np.array([[2]])})
    assert not logger.improved()
    assert logger.best() == 3
    assert logger.latest() == 2
    assert logger.best_epoch() == 2

    logger.update({"a": np.array([[4]])})
    assert logger.improved()
    assert logger.best() == logger.latest() == 4
    assert logger.best_epoch() == 5


def test_none_optimizing_metric():
    logger = log.TrainingLogger(
        optimizing_metric=None, optimizing_mode="min", patience=1
    )
    logger.restart()

    logger.update({"a": np.array([[2]])})
    assert logger.best() == logger.latest() == None
    assert logger.best_epoch() == -1
    assert logger.improved()

    logger.update({"a": np.array([[1]])})
    assert logger.best_epoch() == -1
    assert logger.improved()

    logger.update({"a": np.array([[0]])})
    assert logger.best_epoch() == -1
    assert not logger.early_stop()
    assert logger.improved()


def test_best_epoch_metrics():
    logger = log.TrainingLogger(optimizing_metric="a", optimizing_mode="max")
    val = {"a": np.random.rand(10, 5), "b": np.random.rand(10, 7)}
    train = {"a": np.random.rand(10, 5), "b": np.random.rand(10, 7)}
    test = {"a": np.random.rand(10, 5), "b": np.random.rand(10, 7)}
    val["a"][4] = 2

    best_epoch = logger.best_epoch(val)
    best_metrics_train = logger.best_epoch_metrics(val, train)
    best_metrics_test = logger.best_epoch_metrics(val, test)
    best_metrics_val = logger.best_epoch_metrics(val, val)

    assert best_epoch == 4
    assert best_metrics_train["a"] == train["a"][4].mean()
    assert best_metrics_test["a"] == test["a"][4].mean()
    assert best_metrics_train["b"] == train["b"][4].mean()
    assert best_metrics_test["b"] == test["b"][4].mean()
    assert best_metrics_val["a"] == val["a"][4].mean()
