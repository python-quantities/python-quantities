
from functools import wraps

from nose.tools import assert_equal
import numpy as np
from numpy.testing import Tester, assert_array_equal, assert_array_almost_equal

from .. import markup

USE_UNICODE = False
def setup():
    global USE_UNICODE
    USE_UNICODE = markup.config.use_unicode
    markup.config.use_unicode = False

def teardown():
    markup.config.use_unicode = USE_UNICODE

class QuantityTester(Tester):
    def test(self, *args, **kwargs):
        argv = kwargs.get('extra_argv', [])
        argv.append('--exe')
        kwargs['extra_argv'] = argv
        return Tester.test(self, *args, **kwargs)
test = QuantityTester().test

def assert_quantity_equal(x, y, err_msg='', verbose=True):
    try:
        assert_array_equal(x, y, err_msg, verbose)
        assert_equal(x._dimensionality, y._dimensionality)
    except AssertionError:
        raise AssertionError(
            'Quantities are not equal:\nx: %s\ny: %s'
            % (str(x), str(y))
        )
    except AttributeError:
        if (
            getattr(x, '_dimensionality', None)
            or getattr(y, '_dimensionality', None)
        ):
            raise AssertionError(
                'Quantities are not equal:\nx: %s\ny: %s'
                % (str(x), str(y))
            )

def assert_quantity_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    try:
        assert_array_almost_equal(x, y, decimal, err_msg, verbose)
        assert_equal(x._dimensionality, y._dimensionality)
    except AssertionError:
        raise AssertionError(
            'Quantities are not equal:\nx: %s\ny: %s'
            % (str(x), str(y))
        )
    except AttributeError:
        if (
            getattr(x, '_dimensionality', None)
            or getattr(y, '_dimensionality', None)
        ):
            raise AssertionError(
                'Quantities are not equal:\nx: %s\ny: %s'
                % (str(x), str(y))
            )
