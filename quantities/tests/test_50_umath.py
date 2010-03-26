# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *
from numpy.testing.decorators import skipif as skip_if

import numpy as np
from .. import units
from .. import umath

from . import assert_quantity_equal, assert_quantity_almost_equal


def test_sumproddifffuncs():
    """
    test the sum, product and difference ufuncs
    """
    a = [1,2,3,4] * units.J

    #prod

    assert_array_almost_equal(umath.prod(a), 24 * units.J**4)

    #sum

    assert_almost_equal(umath.sum(a), 10 * units.J)

    #nansum

    c = [1,2,3, np.NaN] * units.kPa
    assert_almost_equal(umath.nansum( c), 6 * units.kPa)

    #cumprod

    assert_raises(ValueError, umath.cumprod, c)

    d = [10, .1, 5, 50] * units.dimensionless
    assert_array_almost_equal(umath.cumprod(d), [10, 1, 5, 250] * units.dimensionless)

    #cumsum

    assert_array_almost_equal(umath.cumsum(a), [1, 3, 6, 10] * units.J)

    assert_array_almost_equal(
        umath.diff([100, 105, 106, 1008] * units.BTU, 1),
        [5, 1, 902] * units.BTU
    )
    assert_array_almost_equal(
        umath.diff([100, 105, 106, 1008] * units.BTU, 2),
        [-4, 901] * units.BTU
    )


    #ediff1d
    y = [1, 1.1 , 1.2, 1.3, 1.3 , 1.3] * units.J / (units.m**2)
    z = [.1, .1, .1, 0,0] * units.J / (units.m**2)

    assert_array_almost_equal(umath.ediff1d(y), z)

    #gradient
    l = umath.gradient([[1,1],[3,4]] * units.J , 1 * units.m)

    assert_array_almost_equal(
        l[0],
        [[2., 3.], [2., 3.]] * units.J/units.m
    )

    assert_array_almost_equal(
        l[1],
        [[0., 0.], [1., 1.]] * units.J/units.m
    )

    #cross

    a = [3,-3, 1] * units.kPa
    b = [4, 9, 2] * units.m**2

    c = umath.cross(a,b)
    assert_array_equal(umath.cross(a,b), [-15, -2, 39] * units.kPa * units.m**2)

    #trapz
    assert_almost_equal(umath.trapz(y, dx = .2 * units.m**2), 1.21 * units.J)

def test_hyperbolicfunctions():
    """
    test the hyperbolic ufuncs
    """
    a = [1, 2, 3, 4, 6] * umath.radian

    assert_array_almost_equal(
        np.sinh(a),
        np.sinh(np.array([1, 2, 3, 4, 6])) * units.dimensionless
    )

    b = [1, 2, 3, 4, 6] * units.dimensionless

    assert_array_almost_equal(
        np.arcsinh(b),
        np.arcsinh(np.array([1, 2, 3, 4, 6])) * units.dimensionless
    )


    assert_array_almost_equal(
        np.cosh(a),
        np.cosh(np.array([1, 2, 3, 4, 6])) * units.dimensionless
    )

    assert_array_almost_equal(
        np.arccosh(b),
        np.arccosh(np.array([1, 2, 3, 4, 6])) * units.dimensionless
    )

    assert_array_almost_equal(
        np.tanh(a),
        np.tanh(np.array([1, 2, 3, 4, 6])) * units.dimensionless
    )

    c = [.01, .5, .6, .8, .99] * units.dimensionless
    assert_array_almost_equal(
        np.arctanh(c),
        np.arctanh(np.array([.01, .5, .6, .8, .99])) * units.dimensionless
    )

def test_rounding():
    """test rounding functions"""
    #test around
    assert_array_almost_equal(
        np.around([.5, 1.5, 2.5, 3.5, 4.5] * units.J) ,
        [0., 2., 2., 4., 4.] * units.J
    )

    assert_array_almost_equal(
        np.around([1,2,3,11] * units.J, decimals=1),
        [1, 2, 3, 11] * units.J
    )

    assert_array_almost_equal(
        np.around([1,2,3,11] * units.J, decimals=-1),
        [0, 0, 0, 10] * units.J
    )

    # round_ and around are equivalent
    assert_array_almost_equal(
        np.round_([.5, 1.5, 2.5, 3.5, 4.5] * units.J),
        [0., 2., 2., 4., 4.] * units.J
    )

    assert_array_almost_equal(
        np.round_([1,2,3,11] * units.J, decimals=1),
        [1, 2, 3, 11] * units.J
    )

    assert_array_almost_equal(
        np.round_([1,2,3,11] * units.J, decimals=-1),
        [0, 0, 0, 10] * units.J
    )

    #test rint
    a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9] * units.m
    assert_array_almost_equal(
        np.rint(a),
        [-4., -4., -2., 0., 2., 3., 4.]*units.m
    )

    # test floor
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * units.m
    assert_array_almost_equal(
        np.floor(a),
        [-2., -2., -1., 0., 1., 1., 2.] * units.m
    )

    # test ceil
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * units.m
    assert_array_almost_equal(
        np.ceil(a),
        [-1., -1., -0., 1., 2., 2., 2.] * units.m
    )

@skip_if(np.__version__[:5] < '1.4.1')
def test_fix():
    assert_array_equal(np.fix(3.14 * units.degF), 3.0 * units.degF)
    assert_array_equal(np.fix(3.0 * units.degF), 3.0 * units.degF)
    assert_array_equal(
        np.fix([2.1, 2.9, -2.1, -2.9] * units.degF),
        [2., 2., -2., -2.] * units.degF
    )


def test_exponents_and_logarithms():
    """
    test exponens and logarithms ufuncs
    """
    a = [12, 3, 4, 5, 6, 7, -1, -10] * units.dimensionless
    b = [1.62754791e+05, 2.00855369e+01, 5.45981500e+01, 1.48413159e+02,
         4.03428793e+02, 1.09663316e+03, 3.67879441e-01,   4.53999298e-05
        ] * units.dimensionless
    assert_array_almost_equal(np.exp(a), b, 3)

    assert_array_almost_equal(a, np.log(b), 8)

    c = [100, 10000, 5, 4, 1] * units.dimensionless

    assert_array_almost_equal(
        np.log10(c),
        [2., 4., 0.69897, 0.60205999, 0.] * units.dimensionless,
        8
    )

    assert_array_almost_equal(
        np.log2(c),
        [6.64385619, 13.28771238, 2.32192809, 2., 0.] * units.dimensionless,
        8
    )

    e = [1e-10, -1e-10, -7e-10, 1, 0, 1e-5] * units.dimensionless

    f = [1.00000000e-10, -1.00000000e-10, -7.00000000e-10,
         1.71828183e+00, 0.00000000e+00, 1.00000500e-05] * units.dimensionless
    assert_array_almost_equal(np.expm1(e), f, 8)

    assert_array_almost_equal(np.log1p(f), e, 8)


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
                self.assertAlmostEqual(af1[ind], af2[ind], places=prec)
            af1, af2 = a1.flat.imag, a2.flat.imag
            for ind in xrange(af1.nelements()):
                self.assertAlmostEqual(af1[ind], af2[ind], places=prec)
        else:
            af1, af2 = a1.flat, a2.flat
            for x1 , x2 in zip(af1, af2):
                self.assertAlmostEqual(x1, x2, places=prec)


    def test_numpy_trig_functions(self):

        #exp
        self.assertAlmostEqual(np.exp(1.2 * units.dimensionless), 3.32011692)

        t1 = [1,1.5,2.0] * units.dimensionless
        self.numAssertAlmostEqual(np.exp(t1), [2.71828183, 4.48168907, 7.3890561] * units.dimensionless, 8)

        # sin
        self.assertAlmostEqual(np.sin( 5 * units.radian), -0.958924275 * units.dimensionless)
        t2 = [1,2,3,4] * units.radian
        self.numAssertAlmostEqual(np.sin(t2) , [0.841470985, 0.909297427, 0.141120008, -0.756802495] * units.dimensionless, 8)

        # arcsin
        self.assertAlmostEqual(np.arcsin( -0.958924275 * units.dimensionless),  -1.28318531 * units.radian)
        t3 = [0.841470985, 0.909297427, 0.141120008, -0.756802495] * units.dimensionless
        self.numAssertAlmostEqual(np.arcsin(t3) , [1,1.14159265,0.141592654,-0.858407346] * units.radian, 8)


        # cos
        self.assertAlmostEqual(np.cos( 5 * units.radian),
                                0.283662185 * units.dimensionless)
        t2 = [1,2,3,4] * units.radian
        self.numAssertAlmostEqual(np.cos(t2) , [0.540302306, -0.416146837,
                                                -0.989992497, -0.653643621]
                                                 * units.dimensionless, 8)

        # arccos
        self.assertAlmostEqual(np.arccos( 0.283662185 * units.dimensionless),
                               1.28318531 * units.radian)
        t3 = [0.540302306, -0.416146837,
              -0.989992497, -0.653643621] * units.dimensionless
        self.numAssertAlmostEqual(np.arccos(t3) ,
                                   [1,2,3,2.28318531] * units.radian, 8)

        # tan
        self.assertAlmostEqual(np.tan( 5 * units.radian),
                               -3.38051501 * units.dimensionless)
        t2 = [1,2,3,4] * units.radian
        self.numAssertAlmostEqual(np.tan(t2) ,
                                  [1.55740772, -2.18503986,
                                   -0.142546543, 1.15782128] * units.dimensionless, 8)

        # arctan
        self.assertAlmostEqual(np.arctan( 0.283662185 * units.dimensionless),
                                 0.276401407 * units.radian)
        t3 = [1.55740772, -2.18503986, -0.142546543, 1.15782128] * units.dimensionless
        self.numAssertAlmostEqual(np.arctan(t3) ,
                                   [1,-1.14159265,-0.141592654,0.858407346] * units.radian, 8)

        #arctan2
        self.assertAlmostEqual(np.arctan2(1 * units.dimensionless,
                                          0.283662185 * units.dimensionless),
                                           1.2943949196743 * units.radian)
        t4 = [1.55740772, -2.18503986, -0.142546543, 1.15782128] * units.dimensionless
        self.numAssertAlmostEqual(np.arctan2([1,1,1,1] * units.dimensionless ,t3) ,
                                [0.57079632815379,2.7123889798199,
                                 1.7123889803119,0.71238898138855] * units.radian, 8)

        #hypot
        self.assertAlmostEqual(umath.hypot(3 * units.m, 4 * units.m),  5 * units.m)
        t5 = [3, 4, 5, 6] * units.J
        self.numAssertAlmostEqual(umath.hypot([1,1,1,1] * units.J,t5) , [3.16227766,4.12310563,5.09901951,6.08276253] * units.J, 8)

        #degrees
        self.assertAlmostEqual(
            np.degrees(6 * units.radians),
            (6 * units.radians).rescale(units.degree)
        )
        self.assertAlmostEqual(
            np.degrees(6 * units.radians).magnitude,
            (6 * units.radians).rescale(units.degree).magnitude
        )
        self.assertRaises(ValueError, np.degrees, t5)

        #radians
        self.assertAlmostEqual(
            np.radians(6 * units.degree),
            (6 * units.degree).rescale(units.radian)
        )
        self.assertAlmostEqual(
            np.radians(6 * units.degree).magnitude,
            (6 * units.degree).rescale(units.radian).magnitude
        )
        self.assertRaises(ValueError, np.radians, t5)

        #unwrap
        t5 = [5, 10, 20, 30, 40] * units.radians
        t6 = [5., 3.71681469, 1.15044408, -1.41592654, -3.98229715] * units.radians
        self.numAssertAlmostEqual(umath.unwrap(t5), t6, 8)
        self.numAssertAlmostEqual(umath.unwrap(t5, discont = np.pi * units.radians ), t6, 8)

