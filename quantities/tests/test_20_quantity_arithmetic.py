# -*- coding: utf-8 -*-

import operator as op
import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as q


def test_scalar_equality():
    assert_true(q.J == q.J)
    assert_true(1*q.J == q.J)
    assert_true(str(1*q.J) == '1.0 J')
    assert_true(q.J == q.kg*q.m**2/q.s**2)

    assert_false(q.J == q.erg)
    assert_false(2*q.J == q.J)
    assert_false(q.J == 2*q.kg*q.m**2/q.s**2)

    assert_false(q.J == q.kg)

def test_scalar_inequality():
    assert_true(q.J != q.erg)
    assert_true(2*q.J != q.J)
    assert_true(str(2*q.J) != str(q.J))
    assert_true(q.J != 2*q.kg*q.m**2/q.s**2)

    assert_false(q.J != q.J)
    assert_false(1*q.J != q.J)
    assert_false(q.J != 1*q.kg*q.m**2/q.s**2)

def test_scalar_comparison():
    assert_true(2*q.J > q.J)
    assert_true(2*q.J > 1*q.J)
    assert_true(1*q.J >= q.J)
    assert_true(1*q.J >= 1*q.J)
    assert_true(2*q.J >= q.J)
    assert_true(2*q.J >= 1*q.J)

    assert_true(0.5*q.J < q.J)
    assert_true(0.5*q.J < 1*q.J)
    assert_true(0.5*q.J <= q.J)
    assert_true(0.5*q.J <= 1*q.J)
    assert_true(1.0*q.J <= q.J)
    assert_true(1.0*q.J <= 1*q.J)

    assert_false(2*q.J < q.J)
    assert_false(2*q.J < 1*q.J)
    assert_false(2*q.J <= q.J)
    assert_false(2*q.J <= 1*q.J)

    assert_false(0.5*q.J > q.J)
    assert_false(0.5*q.J > 1*q.J)
    assert_false(0.5*q.J >= q.J)
    assert_false(0.5*q.J >= 1*q.J)

def test_array_equality():
    assert_false(
        str(q.Quantity([1, 2, 3, 4], 'J')) == str(q.Quantity([1, 22, 3, 44], 'J'))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')) == str(q.Quantity([1, 2, 3, 4], 'J'))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==q.Quantity([1, 22, 3, 44], 'J')) == \
            str(np.array([True, False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==q.Quantity([1, 22, 3, 44], q.J)) == \
            str(np.array([True, False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==[1, 22, 3, 44]*q.J) == \
            str(np.array([True, False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==np.array([1, 22, 3, 44])*q.J) == \
            str(np.array([True, False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==\
            q.Quantity(np.array([1, 22, 3, 44]), 'J')) == \
            str(np.array([True, False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==\
            q.Quantity(q.Quantity([1, 22, 3, 44], 'J'))) == \
            str(np.array([True, False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==[1, 22, 3, 44]*q.kg*q.m**2/q.s**2) == \
            str(np.array([True, False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')==q.Quantity([1, 22, 3, 44], 'J')) == \
            str(np.array([True, False, True, False]))
    )

def test_array_inequality():
    assert_true(
        str(q.Quantity([1, 2, 3, 4], 'J')!=q.Quantity([1, 22, 3, 44], 'J')) == \
            str(np.array([False, True, False, True]))
    )

def test_array_comparison():
    assert_true(
        str(q.Quantity([1, 2, 33], 'J')>q.Quantity([1, 22, 3], 'J')) == \
            str(np.array([False, False, True]))
    )
    assert_true(
        str(q.Quantity([1, 2, 33], 'J')>=q.Quantity([1, 22, 3], 'J')) == \
            str(np.array([True, False, True]))
    )
    assert_true(
        str(q.Quantity([1, 2, 33], 'J')<q.Quantity([1, 22, 3], 'J')) == \
            str(np.array([False, True, False]))
    )
    assert_true(
        str(q.Quantity([1, 2, 33], 'J')<=q.Quantity([1, 22, 3], 'J')) == \
            str(np.array([True, True, False]))
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
        test1 = 1.5 * q.km
        test2 = 1.5 * q.km

        self.assertEqual(test1, test2)

        # test less than and greater than
        self.assertTrue(1.5 * q.km > 2.5 * q.cm)
        self.assertTrue(1.5 * q.km >= 2.5 * q.cm)
        self.assertTrue(not (1.5 * q.km < 2.5 * q.cm))
        self.assertTrue(not (1.5 * q.km <= 2.5 * q.cm))

        self.assertTrue(
            1.5 * q.km != 1.5 * q.cm,
            "unequal quantities are not-not-equal"
        )

    def test_addition(self):
        # arbitrary test of addition
        self.assertAlmostEqual((5.2 * q.eV) + (300.2 * q.eV), 305.4 * q.eV, 5)

        # test of addition using different units
        self.assertAlmostEqual(
            (5 * q.hp + 7.456999e2 * q.W.rescale(q.hp)),
            (6 * q.hp)
        )

        def add_bad_units():
            """just a function that raises an incompatible units error"""
            return (1 * q.kPa) + (5 * q.lb)

        self.assertRaises(ValueError, add_bad_units)

        # add a scalar and an array
        arr = np.array([1,2,3,4,5])
        temp1 = arr * q.rem
        temp2 = 5.5 * q.rem

        self.assertEqual(
            str(temp1 + temp2),
            "[  6.5   7.5   8.5   9.5  10.5] rem"
        )
        self.assertTrue(((arr+5.5) * q.rem == temp1 + temp2).all())

        # with different units
        temp4 = 1e-2 * q.sievert
        self.numAssertAlmostEqual(
            temp1 + temp4.rescale(q.rem),
            temp1 + 1 * q.rem
        )

        # add two arrays
        temp3 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * q.rem

        self.assertEqual(
            str(temp1 + temp3),
            "[  6.5   8.5   8.5   9.5  10.5] rem"
        )
        # two arrays with different units
        temp5 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * q.sievert

        self.assertEqual(
            str(temp1 + temp5.rescale(q.rem)),
            "[  6.5   8.5   8.5   9.5  10.5] rem"
        )

        # in-place addition
        temp1 = 1*q.m
        temp2 = 1*q.m
        temp1+=temp1
        self.assertEqual(str(temp1), str(temp2+temp2))

        temp1 = [1, 2, 3, 4]*q.m
        temp2 = [1, 2, 3, 4]*q.m
        temp1+=temp1
        self.assertEqual(str(temp1), str(temp2+temp2))

        def iadd(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, iadd, 1*q.m, 1)

    def test_substraction(self):
        # arbitrary test of subtraction
        self.assertAlmostEqual((5.2 * q.eV) - (300.2 * q.eV), -295.0 * q.eV)

        # the formatting should be the same
        self.assertEqual(
            str((5.2 * q.J) - (300.2 * q.J)),
            str(-295.0 * q.J)
        )

        # test of subtraction using different units
        self.assertAlmostEqual(
            (5 * q.hp - 7.456999e2 * q.W.rescale(q.hp)),
            (4 * q.hp)
        )

        def subtract_bad_units():
            """just a function that raises an incompatible units error"""
            return (1 * q.kPa) - (5 * q.lb)

        self.assertRaises(ValueError, subtract_bad_units)

        # subtract a scalar and an array
        arr = np.array([1,2,3,4,5])
        temp1 = arr * q.rem
        temp2 = 5.5 * q.rem

        self.assertEqual(str(temp1 - temp2), "[-4.5 -3.5 -2.5 -1.5 -0.5] rem")
        self.numAssertEqual((arr-5.5) * q.rem, temp1 - temp2)

        # with different units
        temp4 = 1e-2 * q.sievert
        self.numAssertAlmostEqual(temp1 - temp4.rescale(q.rem), temp1 - q.rem)

        #subtract two arrays
        temp3 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * q.rem

        self.assertEqual(str(temp1 - temp3), "[-4.5 -4.5 -2.5 -1.5 -0.5] rem")
        #two arrays with different units
        temp5 = np.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * q.sievert

        self.assertEqual(
            str(temp1 - temp5.rescale(q.rem)),
            "[-4.5 -4.5 -2.5 -1.5 -0.5] rem"
        )

        # in-place
        temp1 = 1*q.m
        temp2 = 1*q.m
        temp1-=temp1
        self.assertEqual(str(temp1), str(temp2-temp2))

        temp1 = [1, 2, 3, 4]*q.m
        temp2 = [1, 2, 3, 4]*q.m
        temp1-=temp1
        self.assertEqual(str(temp1), str(temp2-temp2))

        def isub(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, isub, temp1, 1)

    def test_multiplication(self):
        #arbitrary test of multiplication
        self.assertAlmostEqual(
            (10.3 * q.kPa) * (10 * q.inch),
            103.0 * q.kPa*q.inch
        )

        self.assertAlmostEqual((5.2 * q.J) * (300.2 * q.J), 1561.04 * q.J**2)

        # the formatting should be the same
        self.assertEqual(
            str((10.3 * q.kPa) * (10 * q.inch)),
            str( 103.0 * q.kPa*q.inch)
        )
        self.assertEqual(
            str((5.2 * q.J) * (300.2 * q.J)),
            str(1561.04 * q.J**2)
        )

        # does multiplication work with arrays?
        # multiply an array with a scalar
        temp1  = np.array ([3,4,5,6,7]) * q.J
        temp2 = .5 * q.s**-1

        self.assertEqual(
            str(temp1 * temp2),
            "[ 1.5  2.   2.5  3.   3.5] J/s"
        )

        # multiply an array with an array
        temp3 = np.array ([4,4,5,6,7]) * q.s**-1
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
            (10.3 * q.kPa) / (1 * q.inch),
            10.3 * q.kPa/q.inch
        )

        self.assertAlmostEqual(
            (5.2 * q.eV) / (400.0 * q.eV),
            q.Quantity(.013)
        )

        # the formatting should be the same
        self.assertEqual(
            str((5.2 * q.J) / (400.0 * q.J)),
            str(q.Quantity(.013))
        )

        # divide an array with a scalar
        temp1  = np.array ([3,4,5,6,7]) * q.J
        temp2 = .5 * q.s**-1

        self.assertEqual(
            str(temp1 / temp2),
            "[  6.   8.  10.  12.  14.] s·J"
        )

        # divide an array with an array
        temp3 = np.array([4,4,5,6,7]) * q.s**-1
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
        self.assertAlmostEqual((5.5 * q.cm)**5, (5.5**5) * (q.cm**5))
        self.assertEqual(str((5.5 * q.cm)**5), str((5.5**5) * (q.cm**5)))

        # must also work with compound units
        self.assertAlmostEqual((5.5 * q.J)**5, (5.5**5) * (q.J**5))
        self.assertEqual(str((5.5 * q.J)**5), str((5.5**5) * (q.J**5)))

        # does powering work with arrays?
        temp = np.array([1, 2, 3, 4, 5]) * q.kg
        temp2 = (np.array([1, 8, 27, 64, 125]) **2) * q.kg**6

        self.assertEqual(
            str(temp**3),
            "[   1.    8.   27.   64.  125.] kg³"
        )
        self.assertEqual(str(temp**6), str(temp2))

        def q_pow_r(q1, q2):
            return q1 ** q2

        self.assertRaises(ValueError, q_pow_r, 10.0 * q.m, 10 * q.J)
        self.assertRaises(ValueError, q_pow_r, 10.0 * q.m, np.array([1, 2, 3]))

        self.assertEqual( (10 * q.J) ** (2 * q.J/q.J) , 100 * q.J**2 )

        # test rpow here
        self.assertRaises(ValueError, q_pow_r, 10.0, 10 * q.J)

        self.assertEqual(10**(2*q.J/q.J), 100)

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
        self.assertRaises(ValueError, ipow, 1*q.m, [1, 2])


if __name__ == "__main__":
    run_module_suite()
