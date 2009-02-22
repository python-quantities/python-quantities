# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as q

def num_assert_equal( a1, a2):
        """Test for equality of numarray fields a1 and a2.
        """
        assert_equal(a1.shape, a2.shape)
        assert_equal(a1.dtype, a2.dtype)
        assert_true((a1 == a2).all())

def num_assert_almost_equal( a1, a2, prec = None):
        """Test for approximately equality of numarray fields a1 and a2.
        """
        assert_equal(a1.shape, a2.shape)
        assert_equal(a1.dtype, a2.dtype)

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
                assert_almost_equal(af1[ind], af2[ind], prec)
            af1, af2 = a1.flat.imag, a2.flat.imag
            for ind in xrange(af1.nelements()):
                assert_almost_equal(af1[ind], af2[ind], prec)
        else:
            af1, af2 = a1.flat, a2.flat
            for x1 , x2 in zip(af1, af2):
                assert_almost_equal(x1, x2, prec)

def test_sumproddifffuncs():
    """
    test the sum, product and difference ufuncs
    """
    a = [1,2,3,4] * q.J

    #prod

    num_assert_almost_equal(q.prod(a), 24 * q.J*q.m)

    #sum

    assert_almost_equal(q.sum(a), 10 * q.J)

    #nansum

    c = [1,2,3, np.NaN] * q.kPa
    assert_almost_equal(q.nansum( c), 6 * q.kPa)

    #cumprod

    assert_raises(ValueError, q.cumprod, c)

    d = [10, .1, 5, 50] * q.dimensionless
    num_assert_almost_equal(q.cumprod(d), [10, 1, 5, 250] * q.dimensionless)

    #cumsum

    num_assert_almost_equal(q.cumsum(a), [1, 3, 6, 10] * q.J)

    num_assert_almost_equal(
        q.diff([100, 105, 106, 1008] * q.BTU, 1),
        [5, 1, 902] * q.BTU
    )
    num_assert_almost_equal(
        q.diff([100, 105, 106, 1008] * q.BTU, 2),
        [-4, 901] * q.BTU
    )


    #ediff1d
    y = [1, 1.1 , 1.2, 1.3, 1.3 , 1.3] * q.J / (q.m**2)
    z = [.1, .1, .1, 0,0] * q.J / (q.m**2)

    num_assert_almost_equal(q.ediff1d(y), z)

    #gradient
    l = q.gradient([[1,1],[3,4]] * q.J , 1 * q.m)

    num_assert_almost_equal(
        l[0],
        [[2., 3.], [2., 3.]] * q.J/q.m
    )

    num_assert_almost_equal(
        l[1],
        [[0., 0.], [1., 1.]] * q.J/q.m
    )

    #cross

    a = [3,-3, 1] * q.kPa
    b = [4, 9, 2] * q.m**2

    c = q.cross(a,b)
    num_assert_equal(q.cross(a,b), [-15, -2, 39] * q.kPa * q.m**2)

    #trapz
    assert_almost_equal(q.trapz(y, dx = .2 * q.m**2), 1.21 * q.J)

def test_hyperbolicfunctions():
    """
    test the hyperbolic ufuncs
    """
    a = [1, 2, 3, 4, 6] * q.radian

    num_assert_almost_equal(
        q.sinh(a),
        np.sinh(np.array([1, 2, 3, 4, 6])) * q.dimensionless
    )

    b = [1, 2, 3, 4, 6] * q.dimensionless

    num_assert_almost_equal(
        q.arcsinh(b),
        np.arcsinh(np.array([1, 2, 3, 4, 6])) * q.dimensionless
    )


    num_assert_almost_equal(
        q.cosh(a),
        np.cosh(np.array([1, 2, 3, 4, 6])) * q.dimensionless
    )

    num_assert_almost_equal(
        q.arccosh(b),
        np.arccosh(np.array([1, 2, 3, 4, 6])) * q.dimensionless
    )

    num_assert_almost_equal(
        q.tanh(a),
        np.tanh(np.array([1, 2, 3, 4, 6])) * q.dimensionless
    )

    c = [.01, .5, .6, .8, .99] * q.dimensionless
    num_assert_almost_equal(
        q.arctanh(c),
        np.arctanh(np.array([.01, .5, .6, .8, .99])) * q.dimensionless
    )

def test_rounding():
    """test rounding unctions"""
    #test around
    num_assert_almost_equal(
        q.around([.5, 1.5, 2.5, 3.5, 4.5] * q.J) ,
        [0., 2., 2., 4., 4.] * q.J
    )

    num_assert_almost_equal(
        q.around([1,2,3,11] * q.BTU, decimals=1),
        [1, 2, 3, 11] * q.J
    )

    num_assert_almost_equal(
        q.around([1,2,3,11] * q.BTU, decimals=-1),
        [0, 0, 0, 10] * q.J
    )

    # round_ and around are equivalent
    num_assert_almost_equal(
        q.round_([.5, 1.5, 2.5, 3.5, 4.5] * q.J),
        [0., 2., 2., 4., 4.] * q.J
    )

    num_assert_almost_equal(
        q.round_([1,2,3,11] * q.BTU, decimals=1),
        [1, 2, 3, 11] * q.J
    )

    num_assert_almost_equal(
        q.round_([1,2,3,11] * q.BTU, decimals=-1),
        [0, 0, 0, 10] * q.J
    )

    #test rint
    a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9] * q.kPa
    num_assert_almost_equal(
        np.rint(a),
        [-4., -4., -2., 0., 2., 3., 4.] * q.kPa
    )

    # test fix

    assert_almost_equal(np.fix(3.14 * q.degF), 3.0 * q.degF)
    assert_almost_equal(np.fix(3.0 * q.degF), 3.0 * q.degF)

    num_assert_almost_equal(
        np.fix([2.1, 2.9, -2.1, -2.9] * q.degF),
        [2., 2., -2., -2.] * q.degF
    )

    # test floor
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * q.degC
    num_assert_almost_equal(
        np.floor(a),
        [-2., -2., -1., 0., 1., 1., 2.] * q.degC
    )

    # test ceil
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * q.degC
    num_assert_almost_equal(
        np.ceil(a),
        [-1., -1., -0., 1., 2., 2., 2.] * q.degC)


def test_exponents_and_logarithms():
    """
    test exponens and logarithms ufuncs
    """
    a = [12, 3, 4, 5, 6, 7, -1, -10] * q.dimensionless
    b = [1.62754791e+05, 2.00855369e+01, 5.45981500e+01, 1.48413159e+02,
         4.03428793e+02, 1.09663316e+03, 3.67879441e-01,   4.53999298e-05
        ] * q.dimensionless
    num_assert_almost_equal(q.exp(a), b, prec=3)

    num_assert_almost_equal(a, q.log(b), prec=8)

    c = [100, 10000, 5, 4, 1] * q.dimensionless

    num_assert_almost_equal(
        q.log10(c),
        [2., 4., 0.69897, 0.60205999, 0.] * q.dimensionless,
        prec=8
    )

    num_assert_almost_equal(
        q.log2(c),
        [6.64385619, 13.28771238, 2.32192809, 2., 0.] * q.dimensionless,
        prec=8
    )

    e = [1e-10, -1e-10, -7e-10, 1, 0, 1e-5] * q.dimensionless

    f = [1.00000000e-10, -1.00000000e-10, -7.00000000e-10,
         1.71828183e+00, 0.00000000e+00, 1.00000500e-05] * q.dimensionless
    num_assert_almost_equal(q.expm1(e), f, prec=8)

    num_assert_almost_equal(q.log1p(f), e, prec=8)


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
        self.assertAlmostEqual(q.exp(1.2 * q.dimensionless), 3.32011692)

        t1 = [1,1.5,2.0] * q.radian
        self.numAssertAlmostEqual(q.exp(t1), [2.71828183, 4.48168907, 7.3890561] * q.dimensionless, 8)

        # sin

        self.assertAlmostEqual(q.sin( 5 * q.radian), -0.958924275 * q.dimensionless)
        t2 = [1,2,3,4] * q.radian
        self.numAssertAlmostEqual(q.sin(t2) , [0.841470985, 0.909297427, 0.141120008, -0.756802495] * q.dimensionless, 8)

        # arcsin
        self.assertAlmostEqual(q.arcsin( -0.958924275 * q.dimensionless),  -1.28318531 * q.radian)
        t3 = [0.841470985, 0.909297427, 0.141120008, -0.756802495] * q.dimensionless
        self.numAssertAlmostEqual(q.arcsin(t3) , [1,1.14159265,0.141592654,-0.858407346] * q.radian, 8)


        # cos

        self.assertAlmostEqual(q.cos( 5 * q.radian),
                                0.283662185 * q.dimensionless)
        t2 = [1,2,3,4] * q.radian
        self.numAssertAlmostEqual(q.cos(t2) , [0.540302306, -0.416146837,
                                                -0.989992497, -0.653643621]
                                                 * q.dimensionless, 8)

        # arccos
        self.assertAlmostEqual(q.arccos( 0.283662185 * q.dimensionless),
                               1.28318531 * q.radian)
        t3 = [0.540302306, -0.416146837,
              -0.989992497, -0.653643621] * q.dimensionless
        self.numAssertAlmostEqual(q.arccos(t3) ,
                                   [1,2,3,2.28318531] * q.radian, 8)

        # tan

        self.assertAlmostEqual(q.tan( 5 * q.radian),
                               -3.38051501 * q.dimensionless)
        t2 = [1,2,3,4] * q.radian
        self.numAssertAlmostEqual(q.tan(t2) ,
                                  [1.55740772, -2.18503986,
                                   -0.142546543, 1.15782128] * q.dimensionless, 8)

        # arctan
        self.assertAlmostEqual(q.arctan( 0.283662185 * q.dimensionless),
                                 0.276401407 * q.radian)
        t3 = [1.55740772, -2.18503986, -0.142546543, 1.15782128] * q.dimensionless
        self.numAssertAlmostEqual(q.arctan(t3) ,
                                   [1,-1.14159265,-0.141592654,0.858407346] * q.radian, 8)
        #arctan2


        self.assertAlmostEqual(q.arctan2(1 * q.dimensionless,
                                          0.283662185 * q.dimensionless),
                                           1.2943949196743 * q.radian)
        t4 = [1.55740772, -2.18503986, -0.142546543, 1.15782128] * q.dimensionless
        self.numAssertAlmostEqual(q.arctan2([1,1,1,1] * q.dimensionless ,t3) ,
                                [0.57079632815379,2.7123889798199,
                                 1.7123889803119,0.71238898138855] * q.radian, 8)


        #hypot

        self.assertAlmostEqual(q.hypot(3 * q.m, 4 * q.m),  5 * q.m)
        t5 = [3, 4, 5, 6] * q.J
        self.numAssertAlmostEqual(q.hypot([1,1,1,1] * q.J,t5) , [3.16227766,4.12310563,5.09901951,6.08276253] * q.J, 8)

        #degrees
        self.assertAlmostEqual(
            np.degrees(6 * q.radians),
            (6 * q.radians).rescale(q.degree)
        )
        self.assertAlmostEqual(
            np.degrees(6 * q.radians).magnitude,
            (6 * q.radians).rescale(q.degree).magnitude
        )
        self.assertRaises(ValueError, np.degrees, t5)

        #radians
        self.assertAlmostEqual(
            np.radians(6 * q.degree),
            (6 * q.degree).rescale(q.radian)
        )
        self.assertAlmostEqual(
            np.radians(6 * q.degree).magnitude,
            (6 * q.degree).rescale(q.radian).magnitude
        )
        self.assertRaises(ValueError, np.radians, t5)

        #unwrap

        t5 = [5, 10, 20, 30, 40] * q.radians
        t6 = [5., 3.71681469, 1.15044408, -1.41592654, -3.98229715] * q.radians

        self.numAssertAlmostEqual( q.unwrap(t5), t6, 8)

        self.numAssertAlmostEqual(q.unwrap(t5, discont = np.pi * q.radians ), t6, 8)



if __name__ == "__main__":
    run_module_suite()
