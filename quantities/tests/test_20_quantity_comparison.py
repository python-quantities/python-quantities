# -*- coding: utf-8 -*-

import operator as op

from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
from .. import units

from . import assert_quantity_equal, assert_quantity_almost_equal


def test_scalar_equality():
    assert_array_equal(units.J == units.J, [True])
    assert_array_equal(1*units.J == units.J, [True])
    assert_array_equal(str(1*units.J) == '1.0 J', [True])
    assert_array_equal(units.J == units.kg*units.m**2/units.s**2, [True])

    assert_array_equal(units.J == units.erg, [False])
    assert_array_equal(2*units.J == units.J, [False])
    assert_array_equal(units.J == 2*units.kg*units.m**2/units.s**2, [False])

    assert_array_equal(units.J == units.kg, [False])

def test_scalar_inequality():
    assert_array_equal(units.J != units.erg, [True])
    assert_array_equal(2*units.J != units.J, [True])
    assert_array_equal(str(2*units.J) != str(units.J), [True])
    assert_array_equal(units.J != 2*units.kg*units.m**2/units.s**2, [True])

    assert_array_equal(units.J != units.J, [False])
    assert_array_equal(1*units.J != units.J, [False])
    assert_array_equal(units.J != 1*units.kg*units.m**2/units.s**2, [False])

def test_scalar_comparison():
    assert_array_equal(2*units.J > units.J, [True])
    assert_array_equal(2*units.J > 1*units.J, [True])
    assert_array_equal(1*units.J >= units.J, [True])
    assert_array_equal(1*units.J >= 1*units.J, [True])
    assert_array_equal(2*units.J >= units.J, [True])
    assert_array_equal(2*units.J >= 1*units.J, [True])

    assert_array_equal(0.5*units.J < units.J, [True])
    assert_array_equal(0.5*units.J < 1*units.J, [True])
    assert_array_equal(0.5*units.J <= units.J, [True])
    assert_array_equal(0.5*units.J <= 1*units.J, [True])
    assert_array_equal(1.0*units.J <= units.J, [True])
    assert_array_equal(1.0*units.J <= 1*units.J, [True])

    assert_array_equal(2*units.J < units.J, [False])
    assert_array_equal(2*units.J < 1*units.J, [False])
    assert_array_equal(2*units.J <= units.J, [False])
    assert_array_equal(2*units.J <= 1*units.J, [False])

    assert_array_equal(0.5*units.J > units.J, [False])
    assert_array_equal(0.5*units.J > 1*units.J, [False])
    assert_array_equal(0.5*units.J >= units.J, [False])
    assert_array_equal(0.5*units.J >= 1*units.J, [False])

def test_array_equality():
    assert_array_equal(
        [1, 2, 3, 4]*units.J == [1, 22, 3, 44]*units.J,
        [1, 0, 1, 0]
    )
    assert_array_equal(
        [1, 2, 3, 4]*units.J == [1, 22, 3, 44]*units.kg,
        [0, 0, 0, 0]
    )
    assert_array_equal(
        [1, 2, 3, 4]*units.J == [1, 22, 3, 44],
        [1, 0, 1, 0]
    )

def test_array_inequality():
    assert_array_equal(
        [1, 2, 3, 4]*units.J != [1, 22, 3, 44]*units.J,
        [0, 1, 0, 1]
    )
    assert_array_equal(
        [1, 2, 3, 4]*units.J != [1, 22, 3, 44]*units.kg,
        [1, 1, 1, 1]
    )
    assert_array_equal(
        [1, 2, 3, 4]*units.J != [1, 22, 3, 44],
        [0, 1, 0, 1]
    )

def test_quantity_less_than():
    assert_array_equal(
        [1, 2, 33]*units.J < [1, 22, 3]*units.J,
        [0, 1, 0]
    )
    assert_array_equal(
        [50, 100, 150]*units.cm < [1, 1, 1]*units.m,
        [1, 0, 0]
    )
    assert_array_equal(
        [1, 2, 33]*units.J < [1, 22, 3],
        [0, 1, 0]
    )
    assert_raises(
        ValueError,
        op.lt,
        [1, 2, 33]*units.J,
        [1, 22, 3]*units.kg,
    )

def test_quantity_less_than_or_equal():
    assert_array_equal(
        [1, 2, 33]*units.J <= [1, 22, 3]*units.J,
        [1, 1, 0]
    )
    assert_array_equal(
        [50, 100, 150]*units.cm <= [1, 1, 1]*units.m,
        [1, 1, 0]
    )
    assert_array_equal(
        [1, 2, 33]*units.J <= [1, 22, 3],
        [1, 1, 0]
    )
    assert_raises(
        ValueError,
        op.le,
        [1, 2, 33]*units.J,
        [1, 22, 3]*units.kg,
    )

def test_quantity_greater_than_or_equal():
    assert_array_equal(
        [1, 2, 33]*units.J >= [1, 22, 3]*units.J,
        [1, 0, 1]
    )
    assert_array_equal(
        [50, 100, 150]*units.cm >= [1, 1, 1]*units.m,
        [0, 1, 1]
    )
    assert_array_equal(
        [1, 2, 33]*units.J >= [1, 22, 3],
        [1, 0, 1]
    )
    assert_raises(
        ValueError,
        op.ge,
        [1, 2, 33]*units.J,
        [1, 22, 3]*units.kg,
    )

def test_quantity_greater_than():
    assert_array_equal(
        [1, 2, 33]*units.J > [1, 22, 3]*units.J,
        [0, 0, 1]
    )
    assert_array_equal(
        [50, 100, 150]*units.cm > [1, 1, 1]*units.m,
        [0, 0, 1]
    )
    assert_array_equal(
        [1, 2, 33]*units.J > [1, 22, 3],
        [0, 0, 1]
    )
    assert_raises(
        ValueError,
        op.gt,
        [1, 2, 33]*units.J,
        [1, 22, 3]*units.kg,
    )
