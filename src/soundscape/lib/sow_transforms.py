from functools import partial
from oryx.core import sow
from oryx.core.interpreters.harvest import call_and_reap

from . import utils


def sow_fn(fn, *, name, tag):
    """
    Sow the output of the function fn.
    """

    def _fn(*args, **kwargs):
        val = fn(*args, **kwargs)
        return sow(val, name=name, tag=tag)

    return _fn


def merge_sows(fns):
    """
    Call a list of sowed functions in a loop.
    Only the first functions's output is returned.
    """

    def _fn(*args, **kwargs):
        y = fns[0](*args, **kwargs)
        for fn in fns[1:]:
            fn(*args, **kwargs)
        return y

    return _fn


def sow_fns(fns, *, names, tag):
    """
    Sow the output of a list of functions.
    Only the first functions's output is returned.
    """

    fns = [sow_fn(fn, name=name, tag=tag) for fn, name in zip(fns, names)]
    return merge_sows(fns)


def tuple_to_sow(fn, *, names, tag):
    """
    Sow the outputs of a function that returns a tuple.
    Only the first element of the tuple is returned.
    """

    def _fn(*args, **kwargs):
        vals = fn(*args, **kwargs)
        y = vals[0]
        for name, val in zip(names, vals):
            sow(val, name=name, tag=tag)
        return y

    return _fn


def sow_to_tuple(fn, position=-1, *, name, tag):
    """
    Harvest the output of a function and insert it into
    the tuple returned by the function.
    """

    def _fn(*args, **kwargs):
        vals, reaps = call_and_reap(fn, tag=tag, allowlist=[name])(*args, **kwargs)
        if type(vals) is not tuple:
            vals = (vals,)
        vals = utils.insert_index(vals, position, reaps[name])
        return vals

    return _fn
