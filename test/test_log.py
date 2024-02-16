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
    logger = log.Logger(["a", "b"])
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
    assert logger.logs == {"a": [], "b": []}
    assert logger.close() == {}


def test_incremental_additions():
    logger = log.Logger(["a", "b"])
    logger.restart()

    logger.update({"a": np.array([1]), "c": None})
    assert_tree_equal(logger.merge(), {"a": np.array([1])})

    logger.update({"a": np.array([2]), "b": np.array([3])})
    assert_tree_equal(logger.merge(), {"a": np.array([1, 2]), "b": np.array([3])})

    logger.update({"b": np.array([5])})
    assert_tree_equal(logger.merge(), {"a": np.array([1, 2]), "b": np.array([3, 5])})


def test_serialize():
    logger = log.Logger(["a", "b"])
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2])})
    logger.update({"a": np.array([3]), "b": np.array([4])})
    logger.update({"a": np.array([5]), "b": np.array([6])})
    serialized = logger.serialized()

    assert serialized == '{"a": [1, 3, 5], "b": [2, 4, 6]}'


def test_pbar():
    logger = log.Logger(["a", "b"], pbar_len=4, pbar_keys=["a"])
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
    logger = log.Logger(["a", "b"], merge_fn="stack")
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
    logger = log.Logger(["a", "b"])
    logger.restart()

    logger.update({"a": np.array([1]), "b": np.array([2])})
    logger.update({"a": np.array([3]), "b": np.array([4])})
    logger.update({"a": np.array([5]), "b": np.array([6])}, prefix="test_")
    results = logger.close()

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
    vals = [1, 12, 2.0, 2.01, 2.000000001, 123.0000001, 123.0123131, np.float32("nan")]
    expected = [
        "     1",
        "    12",
        "2.0000",
        "2.0100",
        "2.0000",
        "123.00",
        "123.01",
        "   nan",
    ]
    formatted = [log.format_digits(np.array([val])[0], 6) for val in vals]
    assert formatted == expected


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
