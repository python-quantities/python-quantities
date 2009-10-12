# -*- coding: utf-8 -*-

import operator as op

from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq

from . import assert_quantity_equal, assert_quantity_almost_equal


def test_scalar_equality():
    assert_array_equal(pq.J == pq.J, [True])
    assert_array_equal(1*pq.J == pq.J, [True])
    assert_array_equal(str(1*pq.J) == '1.0 J', [True])
    assert_array_equal(pq.J == pq.kg*pq.m**2/pq.s**2, [True])

    assert_array_equal(pq.J == pq.erg, [False])
    assert_array_equal(2*pq.J == pq.J, [False])
    assert_array_equal(pq.J == 2*pq.kg*pq.m**2/pq.s**2, [False])

    assert_array_equal(pq.J == pq.kg, [False])

def test_scalar_inequality():
    assert_array_equal(pq.J != pq.erg, [True])
    assert_array_equal(2*pq.J != pq.J, [True])
    assert_array_equal(str(2*pq.J) != str(pq.J), [True])
    assert_array_equal(pq.J != 2*pq.kg*pq.m**2/pq.s**2, [True])

    assert_array_equal(pq.J != pq.J, [False])
    assert_array_equal(1*pq.J != pq.J, [False])
    assert_array_equal(pq.J != 1*pq.kg*pq.m**2/pq.s**2, [False])

def test_scalar_comparison():
    assert_array_equal(2*pq.J > pq.J, [True])
    assert_array_equal(2*pq.J > 1*pq.J, [True])
    assert_array_equal(1*pq.J >= pq.J, [True])
    assert_array_equal(1*pq.J >= 1*pq.J, [True])
    assert_array_equal(2*pq.J >= pq.J, [True])
    assert_array_equal(2*pq.J >= 1*pq.J, [True])

    assert_array_equal(0.5*pq.J < pq.J, [True])
    assert_array_equal(0.5*pq.J < 1*pq.J, [True])
    assert_array_equal(0.5*pq.J <= pq.J, [True])
    assert_array_equal(0.5*pq.J <= 1*pq.J, [True])
    assert_array_equal(1.0*pq.J <= pq.J, [True])
    assert_array_equal(1.0*pq.J <= 1*pq.J, [True])

    assert_array_equal(2*pq.J < pq.J, [False])
    assert_array_equal(2*pq.J < 1*pq.J, [False])
    assert_array_equal(2*pq.J <= pq.J, [False])
    assert_array_equal(2*pq.J <= 1*pq.J, [False])

    assert_array_equal(0.5*pq.J > pq.J, [False])
    assert_array_equal(0.5*pq.J > 1*pq.J, [False])
    assert_array_equal(0.5*pq.J >= pq.J, [False])
    assert_array_equal(0.5*pq.J >= 1*pq.J, [False])

def test_array_equality():
    assert_array_equal(
        [1, 2, 3, 4]*pq.J == [1, 22, 3, 44]*pq.J,
        [1, 0, 1, 0]
    )
    assert_array_equal(
        [1, 2, 3, 4]*pq.J == [1, 22, 3, 44]*pq.kg,
        [0, 0, 0, 0]
    )
    assert_array_equal(
        [1, 2, 3, 4]*pq.J == [1, 22, 3, 44],
        [1, 0, 1, 0]
    )

def test_array_inequality():
    assert_array_equal(
        [1, 2, 3, 4]*pq.J != [1, 22, 3, 44]*pq.J,
        [0, 1, 0, 1]
    )
    assert_array_equal(
        [1, 2, 3, 4]*pq.J != [1, 22, 3, 44]*pq.kg,
        [1, 1, 1, 1]
    )
    assert_array_equal(
        [1, 2, 3, 4]*pq.J != [1, 22, 3, 44],
        [0, 1, 0, 1]
    )

def test_quantity_less_than():
    assert_array_equal(
        [1, 2, 33]*pq.J < [1, 22, 3]*pq.J,
        [0, 1, 0]
    )
    assert_array_equal(
        [50, 100, 150]*pq.cm < [1, 1, 1]*pq.m,
        [1, 0, 0]
    )
    assert_array_equal(
        [1, 2, 33]*pq.J < [1, 22, 3],
        [0, 1, 0]
    )
    assert_raises(
        ValueError,
        op.lt,
        [1, 2, 33]*pq.J,
        [1, 22, 3]*pq.kg,
    )

def test_quantity_less_than_or_equal():
    assert_array_equal(
        [1, 2, 33]*pq.J <= [1, 22, 3]*pq.J,
        [1, 1, 0]
    )
    assert_array_equal(
        [50, 100, 150]*pq.cm <= [1, 1, 1]*pq.m,
        [1, 1, 0]
    )
    assert_array_equal(
        [1, 2, 33]*pq.J <= [1, 22, 3],
        [1, 1, 0]
    )
    assert_raises(
        ValueError,
        op.le,
        [1, 2, 33]*pq.J,
        [1, 22, 3]*pq.kg,
    )

def test_quantity_greater_than_or_equal():
    assert_array_equal(
        [1, 2, 33]*pq.J >= [1, 22, 3]*pq.J,
        [1, 0, 1]
    )
    assert_array_equal(
        [50, 100, 150]*pq.cm >= [1, 1, 1]*pq.m,
        [0, 1, 1]
    )
    assert_array_equal(
        [1, 2, 33]*pq.J >= [1, 22, 3],
        [1, 0, 1]
    )
    assert_raises(
        ValueError,
        op.ge,
        [1, 2, 33]*pq.J,
        [1, 22, 3]*pq.kg,
    )

def test_quantity_greater_than():
    assert_array_equal(
        [1, 2, 33]*pq.J > [1, 22, 3]*pq.J,
        [0, 0, 1]
    )
    assert_array_equal(
        [50, 100, 150]*pq.cm > [1, 1, 1]*pq.m,
        [0, 0, 1]
    )
    assert_array_equal(
        [1, 2, 33]*pq.J > [1, 22, 3],
        [0, 0, 1]
    )
    assert_raises(
        ValueError,
        op.gt,
        [1, 2, 33]*pq.J,
        [1, 22, 3]*pq.kg,
    )


if __name__ == "__main__":
    run_module_suite()
