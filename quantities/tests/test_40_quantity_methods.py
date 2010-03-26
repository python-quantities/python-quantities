# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
from .. import units

from . import assert_quantity_equal, assert_quantity_almost_equal


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

    def test_numpy_functions(self):
        # tolist
        k = [[1, 2, 3, 10], [1, 2, 3, 4]] * units.BTU

        self.assertTrue(
            k.tolist() == \
                [[1.0*units.Btu, 2.0*units.Btu, 3.0*units.Btu, 10.0*units.Btu],
                 [1.0*units.Btu, 2.0*units.Btu, 3.0*units.Btu,  4.0*units.Btu]]
        )

        # sum
        temp1 = [100, -100, 20.00003, 1.5e-4] * units.BTU
        self.assertEqual(temp1.sum(), 20.00018 * units.BTU)

        # fill
        u = [[-100, 5, 6], [1, 2, 3]] * units.m
        u.fill(6 * units.ft)
        self.numAssertEqual(u,[[6, 6, 6], [6, 6, 6]] * units.ft)

        # reshape
        y = [[1, 3, 4, 5], [1, 2, 3, 6]] * units.inch
        self.numAssertEqual(
            y.reshape([1,8]),
            [[1.0, 3, 4, 5, 1, 2, 3, 6]] * units.inch
        )

        # transpose
        self.numAssertEqual(
            y.transpose(),
            [[1, 1], [3, 2], [4, 3], [5, 6]] * units.inch
        )

        # flatten
        self.numAssertEqual(
            y.flatten(),
            [1, 3, 4, 5, 1, 2, 3, 6] * units.inch
        )

        # ravel
        self.numAssertEqual(
            y.ravel(),
            [1, 3, 4, 5, 1, 2, 3, 6] * units.inch
        )

        # squeeze
        self.numAssertEqual(
            y.reshape([1,8]).squeeze(),
            [1, 3, 4, 5, 1, 2, 3, 6] * units.inch
        )

        # take
        self.numAssertEqual(
            temp1.take([2, 0, 3]),
            [20.00003, 100, 1.5e-4] * units.BTU
        )

        # put
        # make up something similar to y
        z = [[1, 3, 10, 5], [1, 2, 3, 12]] * units.inch
        # put replace the numbers so it is identical to y
        z.put([2, 7], [4, 6] * units.inch)
        # make sure they are equal
        self.numAssertEqual(z, y)

        # test that the appropriate error is raised
        # when incorrect units are passed
        self.assertRaises(
            ValueError,
            z.put,
            [2, 7], [4, 6] * units.ft
        )
        self.assertRaises(
            TypeError,
            z.put,
            [2, 7], [4, 6]
        )

        # repeat
        z = [1, 1, 1, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 1, 1, 2, 2, 2, 3, 3, 3, \
             6, 6, 6] * units.inch
        self.numAssertEqual(y.repeat(3), z)

        # sort
        m = [4, 5, 2, 3, 1, 6] * units.radian
        m.sort()
        self.numAssertEqual(m, [1, 2, 3, 4, 5, 6] * units.radian)

        # argsort
        m = [1, 4, 5, 6, 2, 9] * units.MeV
        self.numAssertEqual(m.argsort(), np.array([0, 4, 1, 2, 3, 5]))

        # diagonal
        t = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * units.kPa
        self.numAssertEqual(t.diagonal(offset=1), [2, 3] * units.kPa)

        # compress
        self.numAssertEqual(z.compress(z > 5 * units.inch), [6, 6, 6] * units.inch)

        # searchsorted
        m.sort()
        self.numAssertEqual(m.searchsorted([5.5, 9.5] * units.MeV),
                            np.array([4,6]))

        def searchsortedError():
            m.searchsorted([1])

        # make sure the proper error is raised when called with improper units
        self.assertRaises(ValueError, searchsortedError)

        # nonzero
        j = [1, 0, 5, 6, 0, 9] * units.MeV
        self.numAssertEqual(j.nonzero()[0], np.array([0, 2, 3, 5]))

        # max
        self.assertEqual(j.max(), 9 * units.MeV)

        # argmax
        self.assertEqual(j.argmax(), 5)

        # min
        self.assertEqual(j.min(), 0 * units.MeV)

        # argmin
        self.assertEqual(m.argmin(), 0)

        # ptp
        self.assertEqual(m.ptp(), 8 * units.MeV)

        # clip
        self.numAssertEqual(
            j.clip(max=5*units.MeV),
            [1, 0, 5, 5, 0, 5] * units.MeV
        )
        self.numAssertEqual(
            j.clip(min=1*units.MeV),
            [1, 1, 5, 6, 1, 9] * units.MeV
        )
        self.numAssertEqual(
            j.clip(min=1*units.MeV, max=5*units.MeV),
            [1, 1, 5, 5, 1, 5] * units.MeV
        )
        self.assertRaises(ValueError, j.clip)
        self.assertRaises(ValueError, j.clip, 1)

        # round
        p = [1, 3.00001, 3, .6, 1000] * units.J
        self.numAssertEqual(p.round(0), [1, 3., 3, 1, 1000] * units.J)
        self.numAssertEqual(p.round(-1), [0, 0, 0, 0, 1000] * units.J)
        self.numAssertEqual(p.round(3), [1, 3., 3, .6, 1000] * units.J)

        # trace
        d = [[1., 2., 3., 4.], [1., 2., 3., 4.],[1., 2., 3., 4.]]*units.A
        self.numAssertEqual(d.trace(), (1+2+3) * units.A)

        # cumsum
        self.numAssertEqual(
            p.cumsum(),
            [1, 4.00001, 7.00001, 1 + 3.00001 + 3 + .6, 1007.60001] * units.J
        )

        # mean
        self.assertEqual(p.mean(), 201.520002 * units.J)

        # var
        self.assertAlmostEqual(
            p.var(),
            ((1 - 201.520002)**2 + (3.00001 -201.520002)**2 + \
                (3- 201.520002)**2 + (.6 - 201.520002) **2 + \
                (1000-201.520002)**2) / 5 * units.J**2
        )

        # std
        self.assertAlmostEqual(
            p.std(),
            np.sqrt(((1 - 201.520002)**2 + (3.00001 -201.520002)**2 + \
                (3- 201.520002) **2 + (.6 - 201.520002) **2 + \
                (1000-201.520002)**2) / 5) * units.J
            )

        # prod
        o = [1, 3, 2] * units.kPa
        self.assertEqual(o.prod(), 6 * units.kPa**3)

        # cumprod
        self.assertRaises(ValueError, o.cumprod)

        f = [1, 2, 3] * units.dimensionless
        self.numAssertEqual(f.cumprod(), [1,2,6] * units.dimensionless)

        #test conj and conjugate()
        w1 = [1 - 5j, 3 , 6 + 1j] * units.MeV
        self.numAssertEqual(
            w1.conj().magnitude,
            np.array([1-5j, 3, 6+1j]).conj()
        )
        self.numAssertEqual(
            w1.conjugate().magnitude,
            np.array([1-5j, 3, 6+1j]).conjugate()
        )
