# -*- coding: utf-8 -*-

import operator
import os

from nose.tools import *
from numpy.testing import *

import quantities as q
from quantities.dimensionality import Dimensionality


meter = Dimensionality([(q.m, 1)])
meter_repr = 'Dimensionality([(meter, 1)])'
meter_str = 'm'
joule = Dimensionality([(q.kg, 1), (q.m, 2), (q.s, -2)])
joule_repr = 'Dimensionality([(kilogram, 1), (meter, 2), (second, -2)])'
joule_str = 'kg*m**2/s**2'
joule_uni = 'kg·m²/s²'
Joule = Dimensionality([(q.J, 1)])
Joule_repr = 'Dimensionality([(joule, 1)])'
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

def test_addition():
    assert_equal(repr(meter+meter), meter_repr)
    assert_true(meter + meter is not meter)
    assert_equal(repr(meter), meter_repr)
    assert_raises(ValueError, operator.__add__, meter, joule)
    assert_raises(ValueError, operator.__add__, Joule, joule)

def test_subtraction():
    assert_equal(repr(meter-meter), meter_repr)
    assert_true(meter - meter is not meter)
    assert_equal(repr(meter), meter_repr)
    assert_raises(ValueError, operator.__sub__, meter, joule)
    assert_raises(ValueError, operator.__sub__, Joule, joule)


def test_simplification():
    assert_equal(Joule.simplified.string, 'kg*m**2/s**2')

def test_gt():
    assert_true(joule > meter)
    assert_true(Joule > meter)
    assert_false(meter > joule)
    assert_false(meter > Joule)
    assert_false(joule > joule)
    assert_false(joule > Joule)

def test_ge():
    assert_true(joule >= meter)
    assert_true(Joule >= meter)
    assert_false(meter >= joule)
    assert_false(meter >= Joule)
    assert_true(joule >= joule)
    assert_true(joule >= Joule)



if __name__ == "__main__":
    run_module_suite()
