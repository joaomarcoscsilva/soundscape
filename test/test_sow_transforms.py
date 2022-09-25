from oryx.core import reap
import pytest

from soundscape.lib import sow_transforms

f = lambda x: x + 1
g = lambda x: x + 2


def test_sow_fn():
    fn = sow_transforms.sow_fn(f, name="name", tag="tag")

    assert fn(1) == f(1)

    fn = reap(fn, tag="tag")

    assert fn(1) == {"name": f(1)}


def test_merge_sows():
    sowed_f = sow_transforms.sow_fn(f, name="f", tag="tag")
    sowed_g = sow_transforms.sow_fn(g, name="g", tag="tag")

    fn = sow_transforms.merge_sows([sowed_f, sowed_g])

    assert fn(1) == f(1)

    fn = reap(fn, tag="tag")

    assert fn(1) == {"f": f(1), "g": g(1)}


def test_sow_fns():
    fn = sow_transforms.sow_fns([f, g], names=["f", "g"], tag="tag")

    assert fn(1) == f(1)

    fn = reap(fn, tag="tag")

    assert fn(1) == {"f": f(1), "g": g(1)}


def test_tuple_to_sow():
    h = lambda x: (f(x), g(x))
    fn = sow_transforms.tuple_to_sow(h, names=["f", "g"], tag="tag")

    assert fn(1) == f(1)

    fn = reap(fn, tag="tag")
    assert fn(1) == {"f": f(1), "g": g(1)}


def test_sow_to_tuple():
    sowed_f = sow_transforms.sow_fn(f, name="f", tag="tag")
    sowed_g = sow_transforms.sow_fn(g, name="g", tag="tag")

    h = sow_transforms.merge_sows([sowed_f, sowed_g])

    fn = sow_transforms.sow_to_tuple(h, position=1, name="g", tag="tag")

    assert fn(1) == (f(1), g(1))
