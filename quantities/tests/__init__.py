
from functools import wraps

from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .. import markup

markup.config.use_unicode = False


def assert_quantity_equal(x, y, err_msg='', verbose=True):
    try:
        assert_array_equal(x, y, err_msg, verbose)
        assert_equal(x._dimensionality, y._dimensionality)
    except (AssertionError, AttributeError):
        raise AssertionError(
            'Quantities are not equal:\nx: %s\ny: %s'
            % (str(x), str(y))
        )

def assert_quantity_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    try:
        assert_array_almost_equal(x, y, decimal, err_msg, verbose)
        assert_equal(x._dimensionality, y._dimensionality)
    except:
        raise AssertionError(
            'Quantities are not almost equal:\nx: %s\ny: %s'
            % (str(x), str(y))
        )
