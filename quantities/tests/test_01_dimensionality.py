# -*- coding: utf-8 -*-

import operator
import os

from nose.tools import *
from numpy.testing import *

import quantities as pq
from quantities.dimensionality import Dimensionality


meter = Dimensionality({pq.m: 1})
meter_repr = 'Dimensionality({meter: 1})'
meter_str = 'm'
centimeter = Dimensionality({pq.cm: 1})
centimeter_repr = 'Dimensionality({centimeter: 1})'
centimeter_str = 'cm'
joule = Dimensionality({pq.kg: 1, pq.m: 2, pq.s: -2})
joule_repr = 'Dimensionality({kilogram: 1, second: -2, meter: 2})'
joule_str = 'kg*m**2/s**2'
joule_uni = 'kg·m²/s²'
Joule = Dimensionality({pq.J: 1})
Joule_repr = 'Dimensionality({joule: 1})'
Joule_str = 'J'


def test_dimensionality_repr():
    assert_equal(repr(meter), meter_repr)
    assert_equal(repr(joule), joule_repr)
    assert_equal(repr(Joule), Joule_repr)

def test_dimensionality_str():
    assert_equal(str(meter), meter_str)
    assert_equal(joule.string, joule_str)
    assert_equal(joule.unicode, joule_uni)
    assert_equal(Joule.string, 'J')

def test_equality():
    assert_true(meter == meter)
    assert_true(joule == joule)
    assert_false(meter == Joule)
    assert_false(joule == Joule)
    assert_equal(repr(joule), joule_repr)

def test_inequality():
    assert_false(meter != meter)
    assert_false(joule != joule)
    assert_true(meter != Joule)
    assert_true(joule != Joule)
    assert_equal(repr(joule), joule_repr)

def test_copy():
    temp = meter.copy()
    assert_true(temp is not meter)
    assert_true(isinstance(temp, Dimensionality))
    assert_true(temp == meter)
    temp[pq.m] += 1
    assert_false(temp == meter)

def test_addition():
    assert_equal(repr(meter+meter), meter_repr)
    assert_true(meter + meter is not meter)
    assert_raises(ValueError, operator.__add__, meter, joule)
    assert_raises(ValueError, operator.__add__, Joule, joule)
    assert_raises(TypeError, operator.__add__, Joule, 0)
    assert_raises(TypeError, operator.__add__, 0, joule)
    test_dimensionality_repr()

def test_inplace_addition():
    temp = meter.copy()
    temp += meter
    assert_equal(temp, meter)
    assert_raises(ValueError, operator.__iadd__, meter, joule)
    assert_raises(ValueError, operator.__iadd__, Joule, joule)
    assert_raises(TypeError, operator.__iadd__, Joule, 0)
    assert_raises(TypeError, operator.__iadd__, 0, joule)
    test_dimensionality_repr()

def test_subtraction():
    assert_equal(repr(meter-meter), meter_repr)
    assert_true(meter - meter is not meter)
    assert_raises(ValueError, operator.__sub__, meter, joule)
    assert_raises(ValueError, operator.__sub__, Joule, joule)
    assert_raises(TypeError, operator.__sub__, Joule, 0)
    assert_raises(TypeError, operator.__sub__, 0, joule)
    test_dimensionality_repr()

def test_inplace_subtraction():
    temp = meter.copy()
    temp -= meter
    assert_equal(temp, meter)
    assert_raises(ValueError, operator.__isub__, meter, joule)
    assert_raises(ValueError, operator.__isub__, Joule, joule)
    assert_raises(TypeError, operator.__isub__, Joule, 0)
    assert_raises(TypeError, operator.__isub__, 0, joule)
    test_dimensionality_repr()

def test_multiplication():
    assert_equal(meter*meter, Dimensionality({pq.m: 2}))
    assert_equal(meter*centimeter, Dimensionality({pq.m: 1, pq.cm: 1}))
    assert_equal(joule*meter, Dimensionality({pq.kg: 1, pq.m: 3, pq.s: -2}))
    assert_raises(TypeError, operator.__mul__, Joule, 0)
    assert_raises(TypeError, operator.__mul__, 0, joule)
    test_dimensionality_repr()

def test_inplace_multiplication():
    temp = meter.copy()
    temp *= meter
    assert_equal(temp, meter*meter)
    temp *= centimeter
    assert_equal(temp, meter*meter*centimeter)
    temp *= centimeter**-1
    assert_equal(temp, meter*meter)
    assert_raises(TypeError, operator.__imul__, Joule, 0)
    assert_raises(TypeError, operator.__imul__, 0, joule)
    test_dimensionality_repr()

def test_division():
    assert_equal(meter/meter, Dimensionality())
    assert_equal(joule/meter, Dimensionality({pq.kg: 1, pq.m: 1, pq.s: -2}))
    assert_raises(TypeError, operator.__div__, Joule, 0)
    assert_raises(TypeError, operator.__div__, 0, joule)
    test_dimensionality_repr()

def test_inplace_division():
    temp = meter.copy()
    temp /= meter
    assert_equal(temp, meter/meter)
    temp /= centimeter
    assert_equal(temp, meter/meter/centimeter)
    temp /= centimeter**-1
    assert_equal(temp, meter/meter)
    assert_raises(TypeError, operator.__idiv__, Joule, 0)
    assert_raises(TypeError, operator.__idiv__, 0, joule)
    test_dimensionality_repr()

def test_power():
    assert_equal(meter**2, meter*meter)
    assert_equal(meter**0, Dimensionality())
    assert_equal(joule**2, Dimensionality({pq.kg: 2, pq.m: 4, pq.s: -4}))
    assert_raises(TypeError, operator.__pow__, Joule, joule)
    assert_raises(TypeError, operator.__pow__, joule, Joule)
    assert_equal(meter**-1 == meter**-2, False)
    test_dimensionality_repr()

def test_inplace_power():
    temp = meter.copy()
    temp **= 2
    assert_equal(temp, meter**2)
    temp = joule.copy()
    temp **= 2
    assert_equal(temp, joule**2)
    temp = meter.copy()
    temp **= 0
    assert_equal(temp, Dimensionality())
    assert_raises(TypeError, operator.__ipow__, Joule, joule)
    assert_raises(TypeError, operator.__ipow__, joule, Joule)
    test_dimensionality_repr()

def test_simplification():
    assert_equal(Joule.simplified.string, 'kg*m**2/s**2')
    assert_equal(Joule.simplified, joule)

def test_gt():
    assert_true(joule > meter)
    assert_true(Joule > meter)
    assert_false(meter > joule)
    assert_false(meter > Joule)
    assert_false(joule > joule)
    assert_false(joule > Joule)
    test_dimensionality_repr()

def test_ge():
    assert_true(joule >= meter)
    assert_true(Joule >= meter)
    assert_false(meter >= joule)
    assert_false(meter >= Joule)
    assert_true(joule >= joule)
    assert_true(joule >= Joule)
    test_dimensionality_repr()

def test_lt():
    assert_true(meter < joule)
    assert_true(meter < Joule)
    assert_false(joule < meter)
    assert_false(Joule < meter)
    assert_false(joule < joule)
    assert_false(Joule < joule)
    test_dimensionality_repr()

def test_le():
    assert_true(meter <= joule)
    assert_true(meter <= Joule)
    assert_false(joule <= meter)
    assert_false(Joule <= meter)
    assert_true(joule <= joule)
    assert_true(joule <= Joule)
    test_dimensionality_repr()


if __name__ == "__main__":
    run_module_suite()
