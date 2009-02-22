# -*- coding: utf-8 -*-

import operator as op
import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq


def test_scalar_equality():
    assert_true(pq.J == pq.J)
    assert_true(1*pq.J == pq.J)
    assert_true(str(1*pq.J) == '1.0 J')
    assert_true(pq.J == pq.kg*pq.m**2/pq.s**2)

    assert_false(pq.J == pq.erg)
    assert_false(2*pq.J == pq.J)
    assert_false(pq.J == 2*pq.kg*pq.m**2/pq.s**2)

    assert_false(pq.J == pq.kg)

def test_scalar_inequality():
    assert_true(pq.J != pq.erg)
    assert_true(2*pq.J != pq.J)
    assert_true(str(2*pq.J) != str(pq.J))
    assert_true(pq.J != 2*pq.kg*pq.m**2/pq.s**2)

    assert_false(pq.J != pq.J)
    assert_false(1*pq.J != pq.J)
    assert_false(pq.J != 1*pq.kg*pq.m**2/pq.s**2)

def test_scalar_comparison():
    assert_true(2*pq.J > pq.J)
    assert_true(2*pq.J > 1*pq.J)
    assert_true(1*pq.J >= pq.J)
    assert_true(1*pq.J >= 1*pq.J)
    assert_true(2*pq.J >= pq.J)
    assert_true(2*pq.J >= 1*pq.J)

    assert_true(0.5*pq.J < pq.J)
    assert_true(0.5*pq.J < 1*pq.J)
    assert_true(0.5*pq.J <= pq.J)
    assert_true(0.5*pq.J <= 1*pq.J)
    assert_true(1.0*pq.J <= pq.J)
    assert_true(1.0*pq.J <= 1*pq.J)

    assert_false(2*pq.J < pq.J)
    assert_false(2*pq.J < 1*pq.J)
    assert_false(2*pq.J <= pq.J)
    assert_false(2*pq.J <= 1*pq.J)

    assert_false(0.5*pq.J > pq.J)
    assert_false(0.5*pq.J > 1*pq.J)
    assert_false(0.5*pq.J >= pq.J)
    assert_false(0.5*pq.J >= 1*pq.J)

def test_array_equality():
    assert_array_equal(
        pq.Quantity([1, 2, 3, 4], 'J') == pq.Quantity([1, 22, 3, 44], 'J'),
        [1, 0, 1, 0]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 3, 4], 'J') == pq.Quantity([1, 22, 3, 44], 'kg'),
        [0, 0, 0, 0]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 3, 4], 'J') == [1, 22, 3, 44],
        [1, 0, 1, 0]
    )

def test_array_inequality():
    assert_array_equal(
        pq.Quantity([1, 2, 3, 4], 'J') != pq.Quantity([1, 22, 3, 44], 'J'),
        [0, 1, 0, 1]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 3, 4], 'J') != pq.Quantity([1, 22, 3, 44], 'kg'),
        [1, 1, 1, 1]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 3, 4], 'J') != [1, 22, 3, 44],
        [0, 1, 0, 1]
    )

def test_array_comparison():
    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') < pq.Quantity([1, 22, 3], 'J'),
        [0, 1, 0]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') < [1, 22, 3],
        [0, 1, 0]
    )
    assert_raises(
        ValueError,
        op.lt,
        pq.Quantity([1, 2, 33], 'J'),
        pq.Quantity([1, 22, 3], 'kg'),
    )

    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') <= pq.Quantity([1, 22, 3], 'J'),
        [1, 1, 0]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') <= [1, 22, 3],
        [1, 1, 0]
    )
    assert_raises(
        ValueError,
        op.le,
        pq.Quantity([1, 2, 33], 'J'),
        pq.Quantity([1, 22, 3], 'kg'),
    )

    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') >= pq.Quantity([1, 22, 3], 'J'),
        [1, 0, 1]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') >= [1, 22, 3],
        [1, 0, 1]
    )
    assert_raises(
        ValueError,
        op.ge,
        pq.Quantity([1, 2, 33], 'J'),
        pq.Quantity([1, 22, 3], 'kg'),
    )

    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') > pq.Quantity([1, 22, 3], 'J'),
        [0, 0, 1]
    )
    assert_array_equal(
        pq.Quantity([1, 2, 33], 'J') > [1, 22, 3],
        [0, 0, 1]
    )
    assert_raises(
        ValueError,
        op.gt,
        pq.Quantity([1, 2, 33], 'J'),
        pq.Quantity([1, 22, 3], 'kg'),
    )

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

    def test_addition(self):
        # arbitrary test of addition
        self.assertAlmostEqual((5.2 * pq.eV) + (300.2 * pq.eV), 305.4 * pq.eV, 5)

        # test of addition using different units
        self.assertAlmostEqual(
            (5 * pq.hp + 7.456999e2 * pq.W.rescale(pq.hp)),
            (6 * pq.hp)
        )

        def add_bad_units():
            """just a function that raises an incompatible units error"""
            return (1 * pq.kPa) + (5 * pq.lb)

        self.assertRaises(ValueError, add_bad_units)

        # add a scalar and an array
        arr = np.array([1,2,3,4,5])
        temp1 = arr * pq.rem
        temp2 = 5.5 * pq.rem

        self.assertEqual(
            str(temp1 + temp2),
            "[  6.5   7.5   8.5   9.5  10.5] rem"
        )
        self.assertTrue(((arr+5.5) * pq.rem == temp1 + temp2).all())

        # with different units
        temp4 = 1e-2 * pq.sievert
        self.numAssertAlmostEqual(
            temp1 + temp4.rescale(pq.rem),
            temp1 + 1 * pq.rem
        )

        # add two arrays
        temp3 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * pq.rem

        self.assertEqual(
            str(temp1 + temp3),
            "[  6.5   8.5   8.5   9.5  10.5] rem"
        )
        # two arrays with different units
        temp5 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * pq.sievert

        self.assertEqual(
            str(temp1 + temp5.rescale(pq.rem)),
            "[  6.5   8.5   8.5   9.5  10.5] rem"
        )

        # in-place addition
        temp1 = 1*pq.m
        temp2 = 1*pq.m
        temp1+=temp1
        self.assertEqual(str(temp1), str(temp2+temp2))

        temp1 = [1, 2, 3, 4]*pq.m
        temp2 = [1, 2, 3, 4]*pq.m
        temp1+=temp1
        self.assertEqual(str(temp1), str(temp2+temp2))

        def iadd(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, iadd, 1*pq.m, 1)

    def test_substraction(self):
        # arbitrary test of subtraction
        self.assertAlmostEqual((5.2 * pq.eV) - (300.2 * pq.eV), -295.0 * pq.eV)

        # the formatting should be the same
        self.assertEqual(
            str((5.2 * pq.J) - (300.2 * pq.J)),
            str(-295.0 * pq.J)
        )

        # test of subtraction using different units
        self.assertAlmostEqual(
            (5 * pq.hp - 7.456999e2 * pq.W.rescale(pq.hp)),
            (4 * pq.hp)
        )

        def subtract_bad_units():
            """just a function that raises an incompatible units error"""
            return (1 * pq.kPa) - (5 * pq.lb)

        self.assertRaises(ValueError, subtract_bad_units)

        # subtract a scalar and an array
        arr = np.array([1,2,3,4,5])
        temp1 = arr * pq.rem
        temp2 = 5.5 * pq.rem

        self.assertEqual(str(temp1 - temp2), "[-4.5 -3.5 -2.5 -1.5 -0.5] rem")
        self.numAssertEqual((arr-5.5) * pq.rem, temp1 - temp2)

        # with different units
        temp4 = 1e-2 * pq.sievert
        self.numAssertAlmostEqual(temp1 - temp4.rescale(pq.rem), temp1 - pq.rem)

        #subtract two arrays
        temp3 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * pq.rem

        self.assertEqual(str(temp1 - temp3), "[-4.5 -4.5 -2.5 -1.5 -0.5] rem")
        #two arrays with different units
        temp5 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * pq.sievert

        self.assertEqual(
            str(temp1 - temp5.rescale(pq.rem)),
            "[-4.5 -4.5 -2.5 -1.5 -0.5] rem"
        )

        # in-place
        temp1 = 1*pq.m
        temp2 = 1*pq.m
        temp1-=temp1
        self.assertEqual(str(temp1), str(temp2-temp2))

        temp1 = [1, 2, 3, 4]*pq.m
        temp2 = [1, 2, 3, 4]*pq.m
        temp1-=temp1
        self.assertEqual(str(temp1), str(temp2-temp2))

        def isub(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, isub, temp1, 1)

    def test_multiplication(self):
        #arbitrary test of multiplication
        self.assertAlmostEqual(
            (10.3 * pq.kPa) * (10 * pq.inch),
            103.0 * pq.kPa*pq.inch
        )

        self.assertAlmostEqual((5.2 * pq.J) * (300.2 * pq.J), 1561.04 * pq.J**2)

        # the formatting should be the same
        self.assertEqual(
            str((10.3 * pq.kPa) * (10 * pq.inch)),
            str( 103.0 * pq.kPa*pq.inch)
        )
        self.assertEqual(
            str((5.2 * pq.J) * (300.2 * pq.J)),
            str(1561.04 * pq.J**2)
        )

        # does multiplication work with arrays?
        # multiply an array with a scalar
        temp1  = np.array ([3,4,5,6,7]) * pq.J
        temp2 = .5 * pq.s**-1

        self.assertEqual(
            str(temp1 * temp2),
            "[ 1.5  2.   2.5  3.   3.5] J/s"
        )

        # multiply an array with an array
        temp3 = np.array ([4,4,5,6,7]) * pq.s**-1
        self.assertEqual(
            str(temp1 * temp3),
            "[ 12.  16.  25.  36.  49.] J/s"
        )

        # in-place
#        temp1 = 1*m
#        temp2 = 1*m
#        temp1 *= temp1
#        self.assertEqual(str(temp1), str(temp2*temp2))
#
#        temp1 = [1, 2, 3, 4]*m
#        temp2 = [1, 2, 3, 4]*m
#        temp1 *= temp1
#        self.assertEqual(str(temp1), str(temp2*temp2))

    def test_division(self):
        #arbitrary test of division
        self.assertAlmostEqual(
            (10.3 * pq.kPa) / (1 * pq.inch),
            10.3 * pq.kPa/pq.inch
        )

        self.assertAlmostEqual(
            (5.2 * pq.eV) / (400.0 * pq.eV),
            pq.Quantity(.013)
        )

        # the formatting should be the same
        self.assertEqual(
            str((5.2 * pq.J) / (400.0 * pq.J)),
            str(pq.Quantity(.013))
        )

        # divide an array with a scalar
        temp1  = np.array ([3,4,5,6,7]) * pq.J
        temp2 = .5 * pq.s**-1

        self.assertEqual(
            str(temp1 / temp2),
            "[  6.   8.  10.  12.  14.] s·J"
        )

        # divide an array with an array
        temp3 = np.array([4,4,5,6,7]) * pq.s**-1
        self.assertEqual(
            str(temp1 / temp3),
            "[ 0.75  1.    1.    1.    1.  ] s·J"
        )

        # in-place
#        temp1 = 1*m
#        temp2 = 1*m
#        temp1 /= temp1
#        self.assertEqual(str(temp1), str(temp2/temp2))

#        temp1 = [1, 2, 3, 4]*m
#        temp2 = [1, 2, 3, 4]*m
#        temp1 /= temp1
#        self.assertEqual(str(temp1), str(temp2/temp2))

    def test_powering(self):
        # test raising a quantity to a power
        self.assertAlmostEqual((5.5 * pq.cm)**5, (5.5**5) * (pq.cm**5))
        self.assertEqual(str((5.5 * pq.cm)**5), str((5.5**5) * (pq.cm**5)))

        # must also work with compound units
        self.assertAlmostEqual((5.5 * pq.J)**5, (5.5**5) * (pq.J**5))
        self.assertEqual(str((5.5 * pq.J)**5), str((5.5**5) * (pq.J**5)))

        # does powering work with arrays?
        temp = np.array([1, 2, 3, 4, 5]) * pq.kg
        temp2 = (np.array([1, 8, 27, 64, 125]) **2) * pq.kg**6

        self.assertEqual(
            str(temp**3),
            "[   1.    8.   27.   64.  125.] kg³"
        )
        self.assertEqual(str(temp**6), str(temp2))

        def q_pow_r(q1, q2):
            return q1 ** q2

        self.assertRaises(ValueError, q_pow_r, 10.0 * pq.m, 10 * pq.J)
        self.assertRaises(ValueError, q_pow_r, 10.0 * pq.m, np.array([1, 2, 3]))

        self.assertEqual( (10 * pq.J) ** (2 * pq.J/pq.J) , 100 * pq.J**2 )

        # test rpow here
        self.assertRaises(ValueError, q_pow_r, 10.0, 10 * pq.J)

        self.assertEqual(10**(2*pq.J/pq.J), 100)

        # in-place
#        temp1 = 1*m
#        temp2 = 1*m
#        temp1 **= 2
#        self.assertEqual(str(temp1), str(temp2*temp2))
#
#        temp1 = [1, 2, 3, 4]*m
#        temp2 = [1, 2, 3, 4]*m
#        temp1 **= 2
#        self.assertEqual(str(temp1), str(temp2*temp2))

        def ipow(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, ipow, 1*pq.m, [1, 2])


if __name__ == "__main__":
    run_module_suite()
