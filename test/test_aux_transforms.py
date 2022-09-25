import pytest

from soundscape.lib import aux_transforms

f = lambda x: x + 1
g = lambda x: x + 2


def test_add_empty_aux():
    fn = aux_transforms.add_empty_aux(f)
    assert fn(1) == (f(1), {})


def test_attach_aux():
    fn = aux_transforms.add_empty_aux(f)
    fn = aux_transforms.attach_aux(fn, "name", g)
    assert fn(1) == (f(1), {"name": g(1)})


def test_returned_value_to_aux():
    fn = lambda x: (f(x), g(x))
    fn = aux_transforms.returned_value_to_aux(fn, "name", 1)
    assert fn(1) == (f(1), {"name": g(1)})


def test_return_value_from_aux():
    fn = lambda x: (f(x), {"name": g(x)})
    fn = aux_transforms.return_value_from_aux(fn, "name")
    assert fn(1) == (f(1), g(1), {})
