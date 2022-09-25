from functools import partial

from . import utils


def add_empty_aux(fn):
    """
    Add an empty auxiliary dict to the output of fn.
    """

    def _fn(*args, **kwargs):
        return fn(*args, **kwargs), {}

    return _fn


def attach_aux(fn, name, aux_fn):
    """
    Take in a function fn that returns an auxiliary dict and add
    the output of aux_fn to that dict.

    If forward is True, the output of fn is also passed to aux_fn.
    """

    def _fn(*args, **kwargs):
        *vals, aux = fn(*args, **kwargs)
        new_val = aux_fn(*args, **kwargs)
        return *vals, {name: new_val, **aux}

    return _fn


def returned_value_to_aux(fn, name, position):
    def _fn(*args, **kwargs):
        vals = fn(*args, **kwargs)
        vals, val = utils.remove_index(vals, position)
        return *vals, {name: val}

    return _fn


def return_value_from_aux(fn, name, position=-1):
    def _fn(*args, **kwargs):
        *vals, aux = fn(*args, **kwargs)
        aux, val = utils.remove_key(aux, name)
        vals = utils.insert_index(vals, position, val)
        return *vals, aux

    return _fn
