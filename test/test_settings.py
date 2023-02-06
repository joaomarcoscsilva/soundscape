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


def test_settings_calls():
    settings.from_dict(
        {
            "a": 1,
            "b": 2,
        }
    )

    assert g() == 2
    assert g(b=1) == 1
    with pytest.raises(TypeError):
        g(1)

    assert f(1) == 2
    assert f(1, a=2) == 3
    assert f(x=1, a=2) == 3

    assert h(1) == 2
    assert h(1, a=2) == 3
    assert h(x=1, a=2) == 3
    with pytest.raises(TypeError):
        h(1, 2)

    assert i() == 3
    assert i(a=2) == 4
    assert i(b=2) == 3
    assert i(a=2, b=2) == 4
