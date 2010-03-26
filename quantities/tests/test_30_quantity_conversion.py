# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
from .. import units
from ..quantity import Quantity

from . import assert_quantity_equal, assert_quantity_almost_equal

def test_quantity_creation():
    assert_raises(LookupError, Quantity, 1, 'nonsense')
    assert_equal(str(Quantity(1, '')), '1 dimensionless')

def test_unit_conversion():
    x = Quantity(10., 'm')
    x.units = units.ft
    assert_quantity_almost_equal(x, 32.80839895 * units.ft)

    x = 10 * units.m
    x.units = units.ft
    assert_quantity_almost_equal(x, 32.80839895 * units.ft)

    x = 10 * units.m
    x.units = u'ft'
    assert_quantity_almost_equal(x, 32.80839895 * units.ft)

def test_compound_reduction():

    pc_per_cc = units.CompoundUnit("pc/cm**3")
    temp = pc_per_cc * units.CompoundUnit('m/m**3')
    assert_equal(str(temp), "1.0 (pc/cm**3)*(m/m**3)")

    temp = temp.simplified
    temp = temp.rescale(units.pc**-4)
    assert_equal(str(temp), "2.79740021556e+88 1/pc**4")

    temp = temp.rescale(units.m**-4)
    assert_equal(str(temp), "3.08568025e+22 1/m**4")
    assert_equal(str(1/temp), "3.24077648681e-23 m**4")
    assert_equal(str(temp**-1), "3.24077648681e-23 m**4")

    # does this handle regular units correctly?
    temp1 = 3.14159 * units.m

    assert_quantity_almost_equal(temp1, temp1.simplified)

    assert_equal(str(temp1), str(temp1.simplified))

def test_default_units():
    units.set_default_units(length='mm')
    assert_equal(units.m.simplified.magnitude, 1000)
    assert_equal(units.m.simplified.units, units.mm)
    x = 1*units.m
    y = x.simplified
    assert_equal(y.magnitude, 1000)
    assert_equal(y.units, units.mm)
    units.set_default_units(length='m')
    assert_equal(units.m.simplified.magnitude, 1)
    assert_equal(units.m.simplified.units, units.m)
    z = y.simplified
    assert_equal(z.magnitude, 1)
    assert_equal(z.units, units.m)

class TestQuantities(unittest.TestCase):

    def numAssertEqual(self, a1, a2):
        """Test for equality of numarray fields a1 and a2.
        """
        self.assertEqual(a1.shape, a2.shape)
        self.assertEqual(a1.dtype, a2.dtype)
        self.assertTrue((a1 == a2).all())

    def numAssertAlmostEqual(self, a1, a2, prec = None):
        """Test for approximately equality of numarray fields a1 and a2.
        """
        self.assertEqual(a1.shape, a2.shape)
        self.assertEqual(a1.dtype, a2.dtype)

        if prec == None:
            if a1.dtype == 'Float64' or a1.dtype == 'Complex64':
                prec = 15
            else:
                prec = 7
        # the complex part of this does not function correctly and will throw
        # errors that need to be fixed if it is to be used
        if np.iscomplex(a1).all():
            af1, af2 = a1.flat.real, a2.flat.real
            for ind in xrange(af1.nelements()):
                self.assertAlmostEqual(af1[ind], af2[ind], prec)
            af1, af2 = a1.flat.imag, a2.flat.imag
            for ind in xrange(af1.nelements()):
                self.assertAlmostEqual(af1[ind], af2[ind], prec)
        else:
            af1, af2 = a1.flat, a2.flat
            for x1 , x2 in zip(af1, af2):
                self.assertAlmostEqual(x1, x2, prec)

    def test_simple(self):
        self.assertEqual(str(units.m), "1 m (meter)", str(units.m))
        self.assertEqual(str(units.J), "1 J (joule)", str(units.J))

    def test_creation(self):
        self.numAssertEqual(
            [100, -1.02, 30] * units.cm**2,
            Quantity(np.array([100, -1.02, 30]),
            units.cm**2)
        )
        self.assertEqual(
            str([100, -1.02, 30] * units.cm**2),
            str(Quantity(np.array([100, -1.02, 30]), units.cm**2))
        )

        self.assertEqual(
            -10.1 * units.ohm,
            Quantity(-10.1, units.ohm)
        )

        self.assertEqual(
            str(-10.1 * units.ohm),
            str(Quantity(-10.1, units.ohm))
        )

    def test_unit_aggregation(self):
        joule = units.kg*units.m**2/units.s**2
        pc_per_cc = units.CompoundUnit("pc/cm**3")
        area_per_volume = units.CompoundUnit("m**2/m**3")
        self.assertEqual(str(joule/units.m), "1.0 kg*m/s**2", str(joule/units.m))
        self.assertEqual(str(joule*units.m), "1.0 kg*m**3/s**2", str(joule*units.m))
        self.assertEqual(
            str(units.J*pc_per_cc),
            "1.0 J*(pc/cm**3)",
            str(units.J*pc_per_cc)
        )
        temp = pc_per_cc / area_per_volume
        self.assertEqual(
            str(temp.simplified),
            "3.08568025e+22 1/m",
            str(temp.simplified)
        )

    def test_ratios(self):
        self.assertAlmostEqual(
            units.m/units.ft.rescale(units.m),
            3.280839895,
            places=10,
            msg=units.m/units.ft.rescale(units.m)
        )
        self.assertAlmostEqual(
            units.J/units.BTU.rescale(units.J),
            0.00094781712,
            places=10,
            msg=units.J/units.BTU.rescale(units.J)
        )

    def test_equality(self):
        test1 = 1.5 * units.km
        test2 = 1.5 * units.km

        self.assertEqual(test1, test2)

        # test less than and greater than
        self.assertTrue(1.5 * units.km > 2.5 * units.cm)
        self.assertTrue(1.5 * units.km >= 2.5 * units.cm)
        self.assertTrue(not (1.5 * units.km < 2.5 * units.cm))
        self.assertTrue(not (1.5 * units.km <= 2.5 * units.cm))

        self.assertTrue(
            1.5 * units.km != 1.5 * units.cm,
            "unequal quantities are not-not-equal"
        )

    def test_getitem(self):
        tempArray1 = Quantity(np.array([1.5, 2.5 , 3, 5]), units.J)
        temp = 2.5 * units.J
        # check to see if quantities brought back from an array are good
        self.assertEqual(tempArray1[1], temp )
        # check the formatting
        self.assertEqual(str(tempArray1[1]), str(temp))

        def tempfunc(index):
            return tempArray1[index]

        # make sure indexing is correct
        self.assertRaises(IndexError, tempfunc, 10)

        # test get item using slicing
        tempArray2 = [100, .2, -1, -5, -6] * units.mA
        tempArray3 = [100, .2, -1, -5] * units.mA
        tempArray4 = [.2, -1 ] * units.mA

        self.numAssertEqual(tempArray2[:], tempArray2)

        self.numAssertEqual(tempArray2[:-1], tempArray3)
        self.numAssertEqual(tempArray2[1:3], tempArray4)

    def test_setitem (self):
        temp = Quantity([0,2,5,7.6], units.lb)

        # needs to check for incompatible units
        def test(value):
            temp[2] = value

        # make sure normal assignment works correctly
        test(2 *units.lb)

        self.assertRaises(ValueError, test, 60 * units.inch * units.J)

        #test set item using slicing
        tempArray2 = [100, .2, -1, -5, -6] * units.mA
        tempArray3 = [100, .2, 0, 0, -6] * units.mA
        tempArray4 = [100,  1,  1,  1,  1] * units.mA

        tempArray4[1:] = [.2, -1, -5, -6] * units.mA
        self.numAssertEqual(tempArray4, tempArray2)

        tempArray3[2:4] = [-1, -5] * units.mA
        self.numAssertEqual(tempArray3, tempArray2)

        tempArray4[:] = [100, .2, -1, -5, -6] * units.mA
        self.numAssertEqual(tempArray4, tempArray2)

        # check and see that dimensionless numbers work correctly
        tempArray5 = Quantity([.2, -3, -5, -9,10])
        tempArray6 = Quantity([.2, -3, 0, 0,11])

        tempArray5[4] = 1 + tempArray5[4]
        tempArray5[2:4] = np.zeros(2)

        self.numAssertEqual(tempArray5, tempArray6)

        # needs to check for out of bounds
        def tempfunc(value):
            temp[10] = value

        self.assertRaises(IndexError, tempfunc, 5 * units.lb)

    def test_iterator(self):
        f = np.array([100, 200, 1, 60, -80])
        x = f * units.kPa

        # make sure the iterator objects have the correct units
        for i in x:
            # currently fails
            self.assertEqual(i.units, units.kPa.units)
