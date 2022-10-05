import pytest

from soundscape import settings


@settings.settings_fn
def f(x, *, a):
    return x + a


@settings.settings_fn
def g(*, b):
    return b


@settings.settings_fn
def h(x):
    return f(x)


@settings.settings_fn
def i(*, a, b):
    return a + b


@settings.settings_fn
def rec(*, a):
    with settings.Settings({"a": a + 1}):
        return i()


def test_settings_calls():
    with settings.Settings(
        {
            "a": 1,
            "b": 2,
        }
    ):
        assert g() == 2
        assert g(b=1) == 1
        with pytest.raises(TypeError):
            g(1)

        assert f(1) == 2
        assert f(1, a=2) == 3

        # Can't pass positional arguments as keywords
        with pytest.raises(TypeError):
            f(x=1, a=2)

        assert h(1) == 2
        assert h(1, a=2) == 3
        with pytest.raises(TypeError):
            h(x=1, a=2)
            h(1, 2)

        assert i() == 3
        assert i(a=2) == 4
        assert i(b=2) == 3
        assert i(a=2, b=2) == 4

        returned_dict = settings.settings_dict()

        assert returned_dict["a"] == 1
        assert returned_dict["b"] == 2


def test_recursive():
    with settings.Settings({"a": 1, "b": 2}):
        assert rec() == 4
        assert rec(a=2) == 5
        assert rec(b=1) == 3
        assert rec(a=2, b=1) == 4
        assert rec() == 4
        assert i() == 3

        with settings.Settings({"b": 0}):
            assert rec() == 2
            assert rec(a=3) == 4
            assert rec(b=1) == 3
            assert rec(a=3, b=1) == 5
            assert rec() == 2
            assert i() == 1

        assert rec() == 4
