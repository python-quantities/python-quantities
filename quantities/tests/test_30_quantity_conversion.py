# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq

from . import assert_quantity_equal, assert_quantity_almost_equal

def test_quantity_creation():
    assert_raises(LookupError, pq.Quantity, 1, 'nonsense')
    assert_equal(str(pq.Quantity(1, '')), '1 dimensionless')

def test_unit_conversion():
    x = pq.Quantity(10., 'm')
    x.units = pq.ft
    assert_quantity_almost_equal(x, 32.80839895 * pq.ft)

    x = 10 * pq.m
    x.units = pq.ft
    assert_quantity_almost_equal(x, 32.80839895 * pq.ft)

def test_compound_reduction():

    pc_per_cc = pq.CompoundUnit("pc/cm**3")
    temp = pc_per_cc * pq.CompoundUnit('m/m**3')
    assert_equal(str(temp), "1.0 (pc/cm**3)*(m/m**3)")

    temp = temp.simplified
    temp = temp.rescale(pq.pc**-4)
    assert_equal(str(temp), "2.79740021556e+88 1/pc**4")

    temp = temp.rescale(pq.m**-4)
    assert_equal(str(temp), "3.08568025e+22 1/m**4")
    assert_equal(str(1/temp), "3.24077648681e-23 m**4")
    assert_equal(str(temp**-1), "3.24077648681e-23 m**4")

    # does this handle regular units correctly?
    temp1 = 3.14159 * pq.m

    assert_quantity_almost_equal(temp1, temp1.simplified)

    assert_equal(str(temp1), str(temp1.simplified))

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
        self.assertEqual(str(pq.m), "1 m (meter)", str(pq.m))
        self.assertEqual(str(pq.J), "1 J (joule)", str(pq.J))

    def test_creation(self):
        self.numAssertEqual(
            [100, -1.02, 30] * pq.cm**2,
            pq.Quantity(np.array([100, -1.02, 30]),
            pq.cm**2)
        )
        self.assertEqual(
            str([100, -1.02, 30] * pq.cm**2),
            str(pq.Quantity(np.array([100, -1.02, 30]), pq.cm**2))
        )

        self.assertEqual(
            -10.1 * pq.ohm,
            pq.Quantity(-10.1, pq.ohm)
        )

        self.assertEqual(
            str(-10.1 * pq.ohm),
            str(pq.Quantity(-10.1, pq.ohm))
        )

    def test_unit_aggregation(self):
        joule = pq.kg*pq.m**2/pq.s**2
        pc_per_cc = pq.CompoundUnit("pc/cm**3")
        area_per_volume = pq.CompoundUnit("m**2/m**3")
        self.assertEqual(str(joule/pq.m), "1.0 kg*m/s**2", str(joule/pq.m))
        self.assertEqual(str(joule*pq.m), "1.0 kg*m**3/s**2", str(joule*pq.m))
        self.assertEqual(
            str(pq.J*pc_per_cc),
            "1.0 J*(pc/cm**3)",
            str(pq.J*pc_per_cc)
        )
        temp = pc_per_cc / area_per_volume
        self.assertEqual(
            str(temp.simplified),
            "3.08568025e+22 1/m",
            str(temp.simplified)
        )

    def test_ratios(self):
        self.assertAlmostEqual(
            pq.m/pq.ft.rescale(pq.m),
            3.280839895,
            10,
            pq.m/pq.ft.rescale(pq.m)
        )
        self.assertAlmostEqual(
            pq.J/pq.BTU.rescale(pq.J),
            0.00094781712,
            10,
            pq.J/pq.BTU.rescale(pq.J)
        )

    def test_equality(self):
        test1 = 1.5 * pq.km
        test2 = 1.5 * pq.km

        self.assertEqual(test1, test2)

        # test less than and greater than
        self.assertTrue(1.5 * pq.km > 2.5 * pq.cm)
        self.assertTrue(1.5 * pq.km >= 2.5 * pq.cm)
        self.assertTrue(not (1.5 * pq.km < 2.5 * pq.cm))
        self.assertTrue(not (1.5 * pq.km <= 2.5 * pq.cm))

        self.assertTrue(
            1.5 * pq.km != 1.5 * pq.cm,
            "unequal quantities are not-not-equal"
        )

    def test_getitem(self):
        tempArray1 = pq.Quantity(np.array([1.5, 2.5 , 3, 5]), pq.J)
        temp = 2.5 * pq.J
        # check to see if quantities brought back from an array are good
        self.assertEqual(tempArray1[1], temp )
        # check the formatting
        self.assertEqual(str(tempArray1[1]), str(temp))

        def tempfunc(index):
            return tempArray1[index]

        # make sure indexing is correct
        self.assertRaises(IndexError, tempfunc, 10)

        # test get item using slicing
        tempArray2 = [100, .2, -1, -5, -6] * pq.mA
        tempArray3 = [100, .2, -1, -5] * pq.mA
        tempArray4 = [.2, -1 ] * pq.mA

        self.numAssertEqual(tempArray2[:], tempArray2)

        self.numAssertEqual(tempArray2[:-1], tempArray3)
        self.numAssertEqual(tempArray2[1:3], tempArray4)

    def test_setitem (self):
        temp = pq.Quantity([0,2,5,7.6], pq.lb)

        # needs to check for incompatible units
        def test(value):
            temp[2] = value

        # make sure normal assignment works correctly
        test(2 *pq.lb)

        self.assertRaises(ValueError, test, 60 * pq.inch * pq.J)
        # even in the case when the quantity has no units
        # (maybe this could go away)
        self.assertRaises(ValueError, test, 60)

        #test set item using slicing
        tempArray2 = [100, .2, -1, -5, -6] * pq.mA
        tempArray3 = [100, .2, 0, 0, -6] * pq.mA
        tempArray4 = [100,  1,  1,  1,  1] * pq.mA

        tempArray4[1:] = [.2, -1, -5, -6] * pq.mA
        self.numAssertEqual(tempArray4, tempArray2)

        tempArray3[2:4] = [-1, -5] * pq.mA
        self.numAssertEqual(tempArray3, tempArray2)

        tempArray4[:] = [100, .2, -1, -5, -6] * pq.mA
        self.numAssertEqual(tempArray4, tempArray2)

        # check and see that dimensionless numbers work correctly
        tempArray5 = pq.Quantity([.2, -3, -5, -9,10])
        tempArray6 = pq.Quantity([.2, -3, 0, 0,11])

        tempArray5[4] = 1 + tempArray5[4]
        tempArray5[2:4] = np.zeros(2)

        self.numAssertEqual(tempArray5, tempArray6)

        # needs to check for out of bounds
        def tempfunc(value):
            temp[10] = value

        self.assertRaises(IndexError, tempfunc, 5 * pq.lb)

    def test_iterator(self):
        f = np.array([100, 200, 1, 60, -80])
        x = f * pq.kPa

        # make sure the iterator objects have the correct units
        for i in x:
            # currently fails
            self.assertEqual(i.units, pq.kPa.units)


if __name__ == "__main__":
    run_module_suite()
