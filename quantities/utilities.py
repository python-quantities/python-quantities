from functools import wraps
import operator

from nose.tools import assert_equal
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def memoize(f, cache={}):
    @wraps(f)
    def g(*args, **kwargs):
        key = (f, tuple(args), frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]
    return g


class with_doc:

    """
    This decorator combines the docstrings of the provided and decorated objects
    to produce the final docstring for the decorated object.
    """

    def __init__(self, method, use_header=True):
        self.method = method
        if use_header:
            self.header = \
    """

    Notes
    -----
    """
        else:
            self.header = ''

    def __call__(self, new_method):
        new_doc = new_method.__doc__
        original_doc = self.method.__doc__
        header = self.header

        if original_doc and new_doc:
            new_method.__doc__ = """
    %s
    %s
    %s
        """ % (original_doc, header, new_doc)

        elif original_doc:
            new_method.__doc__ = original_doc

        return new_method


def assert_quantity_equal(x, y, err_msg='', verbose=True):
    assert_array_equal(x, y, err_msg, verbose)
    try:
        assert_equal(x._dimensionality, y._dimensionality)
    except AttributeError:
        raise AssertionError(
            'Quantities are not equal:\nx: %s\ny: %s'
            % (str(x), str(y))
        )

def assert_quantity_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    assert_array_almost_equal(x, y, decimal, err_msg, verbose)
    assert_equal(x._dimensionality, y._dimensionality)
