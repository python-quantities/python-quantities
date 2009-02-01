# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
import numpy
from quantities import *
import quantities as q
from nose.tools import *

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
        if numpy.iscomplex(a1).all():
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
        test the sum produc and difference ufuncs
    """
    a = [1,2,3,4] * q.J

    #prod

    num_assert_almost_equal(q.prod(a), 24 * q.J*q.m)

    #sum

    assert_almost_equal(q.sum(a), 10 * q.J)


    #nansum

    c = [1,2,3, numpy.NaN] * q.kPa
    assert_almost_equal(q.nansum( c), 6 * q.kPa)


    #cumprod

    assert_raises(ValueError, q.cumprod, c)

    d = [10, .1, 5, 50] * dimensionless
    num_assert_almost_equal(q.cumprod(d), [10, 1, 5, 250] * dimensionless)

    #cumsum

    num_assert_almost_equal(q.cumsum(a), [1,3, 6 , 10] * q.J)


    num_assert_almost_equal(q.diff([100,105, 106,1008] * q.BTU, 1), [  5,   1, 902] * q.BTU)
    num_assert_almost_equal(q.diff([100,105, 106,1008] * q.BTU, 2), [ -4, 901] * q.BTU)


    #ediff1d
    y = [1, 1.1 , 1.2, 1.3, 1.3 , 1.3] * q.J / (q.m**2)
    z = [.1, .1, .1, 0,0]  * q.J / (q.m**2)

    num_assert_almost_equal(q.ediff1d(y), z)



    #gradient
    l = q.gradient([[1,1],[3,4]] * q.J , 1 * q.m)

    num_assert_almost_equal(l[0],
                            [[ 2.,  3.],
                            [ 2.,  3.]] * q.J/q.m )

    num_assert_almost_equal(l[1],
                            [[ 0.,  0.],
                            [ 1.,  1.]] * q.J/q.m )

    #cross

    a = [3,-3, 1] * q.kPa
    b = [4, 9, 2] * q.m**2

    c = q.cross(a,b)
    print c
    num_assert_equal(q.cross(a,b), [-15, - 2, 39] * q.kPa * q.m**2 )

    #trapz

    assert_almost_equal(trapz(y, dx = .2 *m**2), 1.21 * q.J)

def test_hyperbolicfunctions():
    """
        test the hyperbolic ufuncs
    """

    a = [1,2,3,4,6] * q.radian

    num_assert_almost_equal( q.sinh(a), numpy.sinh(numpy.array([1,2,3,4,6])) * q.dimensionless )

    b = [1,2,3, 4, 6] * q.dimensionless

    num_assert_almost_equal( q.arcsinh(b) , numpy.arcsinh(numpy.array([1,2,3,4,6])) * q.dimensionless)


    num_assert_almost_equal( q.cosh(a), numpy.cosh(numpy.array([1,2,3,4,6])) * q.dimensionless )

    num_assert_almost_equal( q.arccosh(b) , numpy.arccosh(numpy.array([1,2,3,4,6])) * q.dimensionless)

    num_assert_almost_equal( q.tanh(a), numpy.tanh(numpy.array([1,2,3,4,6])) * q.dimensionless )

    c = [.01,.5,.6,.8,.99] * q.dimensionless
    num_assert_almost_equal( q.arctanh(c) , numpy.arctanh(numpy.array([.01,.5,.6,.8,.99])) * q.dimensionless)

def test_rounding():

    """test rounding unctions"""

    #test around
    num_assert_almost_equal( q.around([.5, 1.5, 2.5, 3.5, 4.5] * J) ,
                              [ 0.,  2.,  2.,  4.,  4.] * q.J)

    num_assert_almost_equal( q.around([1,2,3,11] * BTU,
                                       decimals=1) , [ 1,  2,  3, 11] * q.J)

    num_assert_almost_equal( q.around([1,2,3,11] * BTU,
                                       decimals=-1) , [ 0,  0,  0, 10] * q.J)

    # round_ and around are equivalent
    num_assert_almost_equal( q.round_([.5, 1.5, 2.5, 3.5,
                                        4.5] * J) ,
                                         [ 0.,  2.,  2.,  4.,  4.] * q.J)

    num_assert_almost_equal( q.round_([1,2,3,11] * BTU,
                                       decimals=1) , [ 1,  2,  3, 11] * q.J)

    num_assert_almost_equal( q.round_([1,2,3,11] * BTU,
                                       decimals=-1) , [ 0,  0,  0, 10] * q.J)


    #test rint
    a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9] * q.kPa
    num_assert_almost_equal(q.rint(a), [-4., -4., -2.,  0.,
                                          2.,  3.,  4.] * q.kPa)

    # test fix

    assert_almost_equal(q.fix(3.14 * q.degF), 3.0 * q.degF)

    assert_almost_equal(q.fix(3.0 * q.degF), 3.0 * q.degF)

    num_assert_almost_equal(q.fix([2.1, 2.9, -2.1, -2.9] * q.degF),
                             [ 2.,  2., -2., -2.] * q.degF)

    # test floor
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * degC
    num_assert_almost_equal( q.floor(a), [-2., -2., -1.,  0.,
                                            1.,  1.,  2.] * degC)

    # test ceil
    a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * degC
    num_assert_almost_equal( q.ceil(a), [-1., -1., -0.,
                                           1.,  2.,  2.,  2.] * degC)


def test_exponents_and_logarithms():

        a = [12,3,4,5,6,7, -1, -10] * dimensionless
        b = [  1.62754791e+05,   2.00855369e+01,   5.45981500e+01,
            1.48413159e+02,   4.03428793e+02,   1.09663316e+03,
            3.67879441e-01,   4.53999298e-05] * dimensionless
        num_assert_almost_equal(q.exp(a), b, prec = 3)

        num_assert_almost_equal(a, q.log(b) , prec = 8)

        c = [100,10000, 5, 4, 1] * dimensionless

        num_assert_almost_equal(q.log10(c), [ 2.        ,  4.        ,
                                               0.69897   ,  0.60205999,
                                                 0.        ] * dimensionless,
                                                  prec = 8)

        num_assert_almost_equal(q.log2(c), [  6.64385619,  13.28771238,
                                            2.32192809, 2.,0.]
                                            * dimensionless, prec = 8)

        e = [1e-10, -1e-10, -7e-10, 1, 0, 1e-5] * dimensionless

        f = [  1.00000000e-10,  -1.00000000e-10,  -7.00000000e-10,
        1.71828183e+00,   0.00000000e+00,   1.00000500e-05] * dimensionless
        num_assert_almost_equal(q.expm1(e), f, prec = 8)

        num_assert_almost_equal(q.log1p(f), e, prec = 8)
