from functools import wraps
import operator

import numpy as np
from numpy.testing.utils import build_err_msg

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


## testing utilities, borrowed from the numpy project

def assert_array_compare(
    comparison, x, y, err_msg='', verbose=True, header=''
):
    from numpy.core import asarray, isnan, any
    x = np.array(x, copy=False, subok=True)
    y = np.array(y, copy=False, subok=True)

    def isnumber(x):
        return x.dtype.char in '?bhilqpBHILQPfdgFDG'

    try:
        cond = (x.shape==() or y.shape==()) or x.shape == y.shape
        if not cond:
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(shapes %s, %s mismatch)' % (x.shape,
                                                                  y.shape),
                                verbose=verbose, header=header,
                                names=('x', 'y'))
            if not cond :
                raise AssertionError(msg)

        if (isnumber(x) and isnumber(y)) and (any(isnan(x)) or any(isnan(y))):
            # Handling nan: we first check that x and y have the nan at the
            # same locations, and then we mask the nan and do the comparison as
            # usual.
            xnanid = isnan(x)
            ynanid = isnan(y)
            try:
                assert_array_equal(xnanid, ynanid)
            except AssertionError:
                msg = build_err_msg([x, y],
                                    err_msg
                                    + '\n(x and y nan location mismatch %s, ' \
                                    '%s mismatch)' % (xnanid, ynanid),
                                    verbose=verbose, header=header,
                                    names=('x', 'y'))
            val = comparison(x[~xnanid], y[~ynanid])
        else:
            val = comparison(x,y)
        if isinstance(val, bool):
            cond = val
            reduced = [0]
        else:
            reduced = val.ravel()
            cond = reduced.all()
            reduced = reduced.tolist()
        if not cond:
            match = 100-100.0*reduced.count(1)/len(reduced)
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(mismatch %s%%)' % (match,),
                                verbose=verbose, header=header,
                                names=('x', 'y'))
            if not cond :
                raise AssertionError(msg)
    except ValueError:
        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=('x', 'y'))
        raise ValueError(msg)

def assert_array_equal(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__eq__, x, y, err_msg=err_msg,
                         verbose=verbose, header='Values are not equal')

def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    from numpy.core import around
    def compare(x, y):
        return around(abs(x-y),decimal) <= 10.0**(-decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Values are not almost equal')
