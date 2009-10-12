# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq

from . import assert_quantity_equal, assert_quantity_almost_equal


def test_sumproddifffuncs():
    """
    test the sum, product and difference ufuncs
    """
    a = [1,2,3,4] * pq.J

    #prod

    assert_array_almost_equal(pq.prod(a), 24 * pq.J**4)

    #sum

    assert_almost_equal(pq.sum(a), 10 * pq.J)

    #nansum

    c = [1,2,3, np.NaN] * pq.kPa
    assert_almost_equal(pq.nansum( c), 6 * pq.kPa)

    #cumprod

    assert_raises(ValueError, pq.cumprod, c)

    d = [10, .1, 5, 50] * pq.dimensionless
    assert_array_almost_equal(pq.cumprod(d), [10, 1, 5, 250] * pq.dimensionless)

    #cumsum

    assert_array_almost_equal(pq.cumsum(a), [1, 3, 6, 10] * pq.J)

    assert_array_almost_equal(
        pq.diff([100, 105, 106, 1008] * pq.BTU, 1),
        [5, 1, 902] * pq.BTU
    )
    assert_array_almost_equal(
        pq.diff([100, 105, 106, 1008] * pq.BTU, 2),
        [-4, 901] * pq.BTU
    )


    #ediff1d
    y = [1, 1.1 , 1.2, 1.3, 1.3 , 1.3] * pq.J / (pq.m**2)
    z = [.1, .1, .1, 0,0] * pq.J / (pq.m**2)

    assert_array_almost_equal(pq.ediff1d(y), z)

    #gradient
    l = pq.gradient([[1,1],[3,4]] * pq.J , 1 * pq.m)

    assert_array_almost_equal(
        l[0],
        [[2., 3.], [2., 3.]] * pq.J/pq.m
    )

    assert_array_almost_equal(
        l[1],
        [[0., 0.], [1., 1.]] * pq.J/pq.m
    )

    #cross

    a = [3,-3, 1] * pq.kPa
    b = [4, 9, 2] * pq.m**2

    c = pq.cross(a,b)
    assert_array_equal(pq.cross(a,b), [-15, -2, 39] * pq.kPa * pq.m**2)

    #trapz
    assert_almost_equal(pq.trapz(y, dx = .2 * pq.m**2), 1.21 * pq.J)

def test_hyperbolicfunctions():
    """
    test the hyperbolic ufuncs
    """
    a = [1, 2, 3, 4, 6] * pq.radian

    assert_array_almost_equal(
        pq.sinh(a),
        np.sinh(np.array([1, 2, 3, 4, 6])) * pq.dimensionless
    )

    b = [1, 2, 3, 4, 6] * pq.dimensionless

    assert_array_almost_equal(
        pq.arcsinh(b),
        np.arcsinh(np.array([1, 2, 3, 4, 6])) * pq.dimensionless
    )


    assert_array_almost_equal(
        pq.cosh(a),
        np.cosh(np.array([1, 2, 3, 4, 6])) * pq.dimensionless
    )

    assert_array_almost_equal(
        pq.arccosh(b),
        np.arccosh(np.array([1, 2, 3, 4, 6])) * pq.dimensionless
    )

    assert_array_almost_equal(
        pq.tanh(a),
        np.tanh(np.array([1, 2, 3, 4, 6])) * pq.dimensionless
    )

    c = [.01, .5, .6, .8, .99] * pq.dimensionless
    assert_array_almost_equal(
        pq.arctanh(c),
        np.arctanh(np.array([.01, .5, .6, .8, .99])) * pq.dimensionless
    )

def test_rounding():
    """test rounding unctions"""
    #test around
    assert_array_almost_equal(
        pq.around([.5, 1.5, 2.5, 3.5, 4.5] * pq.J) ,
        [0., 2., 2., 4., 4.] * pq.J
    )

    assert_array_almost_equal(
        pq.around([1,2,3,11] * pq.J, decimals=1),
        [1, 2, 3, 11] * pq.J
    )

    assert_array_almost_equal(
        pq.around([1,2,3,11] * pq.J, decimals=-1),
        [0, 0, 0, 10] * pq.J
    )

    # round_ and around are equivalent
    assert_array_almost_equal(
        pq.round_([.5, 1.5, 2.5, 3.5, 4.5] * pq.J),
        [0., 2., 2., 4., 4.] * pq.J
    )

    assert_array_almost_equal(
        pq.round_([1,2,3,11] * pq.J, decimals=1),
        [1, 2, 3, 11] * pq.J
    )

    assert_array_almost_equal(
        pq.round_([1,2,3,11] * pq.J, decimals=-1),
        [0, 0, 0, 10] * pq.J
    )

    #test rint
    a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9] * pq.kPa
    assert_array_almost_equal(
        np.rint(a),
        [-4., -4., -2., 0., 2., 3., 4.] * pq.kPa
    )

    # test fix
# TODO: uncomment once np.fix behaves itself
#    assert_array_equal(np.fix(3.14 * pq.degF), 3.0 * pq.degF)
#    assert_array_equal(np.fix(3.0 * pq.degF), 3.0 * pq.degF)
#    assert_array_equal(
#        np.fix([2.1, 2.9, -2.1, -2.9] * pq.degF),
#        [2., 2., -2., -2.] * pq.degF
#    )

    # test floor
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * pq.degC
    assert_array_almost_equal(
        np.floor(a),
        [-2., -2., -1., 0., 1., 1., 2.] * pq.degC
    )

    # test ceil
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * pq.degC
    assert_array_almost_equal(
        np.ceil(a),
        [-1., -1., -0., 1., 2., 2., 2.] * pq.degC)


def test_exponents_and_logarithms():
    """
    test exponens and logarithms ufuncs
    """
    a = [12, 3, 4, 5, 6, 7, -1, -10] * pq.dimensionless
    b = [1.62754791e+05, 2.00855369e+01, 5.45981500e+01, 1.48413159e+02,
         4.03428793e+02, 1.09663316e+03, 3.67879441e-01,   4.53999298e-05
        ] * pq.dimensionless
    assert_array_almost_equal(pq.exp(a), b, 3)

    assert_array_almost_equal(a, pq.log(b), 8)

    c = [100, 10000, 5, 4, 1] * pq.dimensionless

    assert_array_almost_equal(
        pq.log10(c),
        [2., 4., 0.69897, 0.60205999, 0.] * pq.dimensionless,
        8
    )

    assert_array_almost_equal(
        pq.log2(c),
        [6.64385619, 13.28771238, 2.32192809, 2., 0.] * pq.dimensionless,
        8
    )

    e = [1e-10, -1e-10, -7e-10, 1, 0, 1e-5] * pq.dimensionless

    f = [1.00000000e-10, -1.00000000e-10, -7.00000000e-10,
         1.71828183e+00, 0.00000000e+00, 1.00000500e-05] * pq.dimensionless
    assert_array_almost_equal(pq.expm1(e), f, 8)

    assert_array_almost_equal(pq.log1p(f), e, 8)


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


    def test_numpy_trig_functions(self):

        #exp
        self.assertAlmostEqual(pq.exp(1.2 * pq.dimensionless), 3.32011692)

        t1 = [1,1.5,2.0] * pq.radian
        self.numAssertAlmostEqual(pq.exp(t1), [2.71828183, 4.48168907, 7.3890561] * pq.dimensionless, 8)

        # sin

        self.assertAlmostEqual(pq.sin( 5 * pq.radian), -0.958924275 * pq.dimensionless)
        t2 = [1,2,3,4] * pq.radian
        self.numAssertAlmostEqual(pq.sin(t2) , [0.841470985, 0.909297427, 0.141120008, -0.756802495] * pq.dimensionless, 8)

        # arcsin
        self.assertAlmostEqual(pq.arcsin( -0.958924275 * pq.dimensionless),  -1.28318531 * pq.radian)
        t3 = [0.841470985, 0.909297427, 0.141120008, -0.756802495] * pq.dimensionless
        self.numAssertAlmostEqual(pq.arcsin(t3) , [1,1.14159265,0.141592654,-0.858407346] * pq.radian, 8)


        # cos

        self.assertAlmostEqual(pq.cos( 5 * pq.radian),
                                0.283662185 * pq.dimensionless)
        t2 = [1,2,3,4] * pq.radian
        self.numAssertAlmostEqual(pq.cos(t2) , [0.540302306, -0.416146837,
                                                -0.989992497, -0.653643621]
                                                 * pq.dimensionless, 8)

        # arccos
        self.assertAlmostEqual(pq.arccos( 0.283662185 * pq.dimensionless),
                               1.28318531 * pq.radian)
        t3 = [0.540302306, -0.416146837,
              -0.989992497, -0.653643621] * pq.dimensionless
        self.numAssertAlmostEqual(pq.arccos(t3) ,
                                   [1,2,3,2.28318531] * pq.radian, 8)

        # tan

        self.assertAlmostEqual(pq.tan( 5 * pq.radian),
                               -3.38051501 * pq.dimensionless)
        t2 = [1,2,3,4] * pq.radian
        self.numAssertAlmostEqual(pq.tan(t2) ,
                                  [1.55740772, -2.18503986,
                                   -0.142546543, 1.15782128] * pq.dimensionless, 8)

        # arctan
        self.assertAlmostEqual(pq.arctan( 0.283662185 * pq.dimensionless),
                                 0.276401407 * pq.radian)
        t3 = [1.55740772, -2.18503986, -0.142546543, 1.15782128] * pq.dimensionless
        self.numAssertAlmostEqual(pq.arctan(t3) ,
                                   [1,-1.14159265,-0.141592654,0.858407346] * pq.radian, 8)
        #arctan2


        self.assertAlmostEqual(pq.arctan2(1 * pq.dimensionless,
                                          0.283662185 * pq.dimensionless),
                                           1.2943949196743 * pq.radian)
        t4 = [1.55740772, -2.18503986, -0.142546543, 1.15782128] * pq.dimensionless
        self.numAssertAlmostEqual(pq.arctan2([1,1,1,1] * pq.dimensionless ,t3) ,
                                [0.57079632815379,2.7123889798199,
                                 1.7123889803119,0.71238898138855] * pq.radian, 8)


        #hypot

        self.assertAlmostEqual(pq.hypot(3 * pq.m, 4 * pq.m),  5 * pq.m)
        t5 = [3, 4, 5, 6] * pq.J
        self.numAssertAlmostEqual(pq.hypot([1,1,1,1] * pq.J,t5) , [3.16227766,4.12310563,5.09901951,6.08276253] * pq.J, 8)

        #degrees
        self.assertAlmostEqual(
            np.degrees(6 * pq.radians),
            (6 * pq.radians).rescale(pq.degree)
        )
        self.assertAlmostEqual(
            np.degrees(6 * pq.radians).magnitude,
            (6 * pq.radians).rescale(pq.degree).magnitude
        )
        self.assertRaises(ValueError, np.degrees, t5)

        #radians
        self.assertAlmostEqual(
            np.radians(6 * pq.degree),
            (6 * pq.degree).rescale(pq.radian)
        )
        self.assertAlmostEqual(
            np.radians(6 * pq.degree).magnitude,
            (6 * pq.degree).rescale(pq.radian).magnitude
        )
        self.assertRaises(ValueError, np.radians, t5)

        #unwrap

        t5 = [5, 10, 20, 30, 40] * pq.radians
        t6 = [5., 3.71681469, 1.15044408, -1.41592654, -3.98229715] * pq.radians

        self.numAssertAlmostEqual( pq.unwrap(t5), t6, 8)

        self.numAssertAlmostEqual(pq.unwrap(t5, discont = np.pi * pq.radians ), t6, 8)



if __name__ == "__main__":
    run_module_suite()
