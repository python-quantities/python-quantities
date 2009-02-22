# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as q


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
        k = [[1, 2, 3, 10], [1, 2, 3, 4]] * q.BTU

        self.assertTrue(
            k.tolist() == \
                [[1.0*q.Btu, 2.0*q.Btu, 3.0*q.Btu, 10.0*q.Btu],
                 [1.0*q.Btu, 2.0*q.Btu, 3.0*q.Btu,  4.0*q.Btu]]
        )

        # sum
        temp1 = [100, -100, 20.00003, 1.5e-4] * q.BTU
        self.assertEqual(temp1.sum(), 20.00018 * q.BTU)

        # fill
        u = [[-100, 5, 6], [1, 2, 3]] * q.m
        u.fill(6 * q.m)
        self.numAssertEqual(u,[[6, 6, 6], [6, 6, 6]] * q.m)
        # incompatible units:
        self.assertRaises(ValueError, u.fill, [[-100, 5, 6], [1, 2, 3]])

        # reshape
        y = [[1, 3, 4, 5], [1, 2, 3, 6]] * q.inch
        self.numAssertEqual(
            y.reshape([1,8]),
            [[1.0, 3, 4, 5, 1, 2, 3, 6]] * q.inch
        )

        # transpose
        self.numAssertEqual(
            y.transpose(),
            [[1, 1], [3, 2], [4, 3], [5, 6]] * q.inch
        )

        # flatten
        self.numAssertEqual(
            y.flatten(),
            [1, 3, 4, 5, 1, 2, 3, 6] * q.inch
        )

        # ravel
        self.numAssertEqual(
            y.ravel(),
            [1, 3, 4, 5, 1, 2, 3, 6] * q.inch
        )

        # squeeze
        self.numAssertEqual(
            y.reshape([1,8]).squeeze(),
            [1, 3, 4, 5, 1, 2, 3, 6] * q.inch
        )

        # take
        self.numAssertEqual(
            temp1.take([2, 0, 3]),
            [20.00003, 100, 1.5e-4] * q.BTU
        )

        # put
        # make up something similar to y
        z = [[1, 3, 10, 5], [1, 2, 3, 12]] * q.inch
        # put replace the numbers so it is identical to y
        z.put([2, 7], [4, 6] * q.inch)
        # make sure they are equal
        self.numAssertEqual(z, y)

        # test that the appropriate error is raised
        # when incorrect units are passed
        self.assertRaises(
            ValueError,
            z.put,
            [2, 7], [4, 6] * q.ft
        )
        self.assertRaises(
            TypeError,
            z.put,
            [2, 7], [4, 6]
        )

        # repeat
        z = [1, 1, 1, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 1, 1, 2, 2, 2, 3, 3, 3, \
             6, 6, 6] * q.inch
        self.numAssertEqual(y.repeat(3), z)

        # sort
        m = [4, 5, 2, 3, 1, 6] * q.radian
        m.sort()
        self.numAssertEqual(m, [1, 2, 3, 4, 5, 6] * q.radian)

        # argsort
        m = [1, 4, 5, 6, 2, 9] * q.MeV
        self.numAssertEqual(m.argsort(), np.array([0, 4, 1, 2, 3, 5]))

        # diagonal
        t = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * q.kPa
        self.numAssertEqual(t.diagonal(offset=1), [2, 3] * q.kPa)

        # compress
        self.numAssertEqual(z.compress(z > 5 * q.inch), [6, 6, 6] * q.inch)

        # searchsorted
        m.sort()
        self.numAssertEqual(m.searchsorted([5.5, 9.5] * q.MeV),
                            np.array([4,6]))

        def searchsortedError():
            m.searchsorted([1])

        # make sure the proper error is raised when called with improper units
        self.assertRaises(ValueError, searchsortedError)

        # nonzero
        j = [1, 0, 5, 6, 0, 9] * q.MeV
        self.numAssertEqual(j.nonzero()[0], np.array([0, 2, 3, 5]))

        # max
        self.assertEqual(j.max(), 9 * q.MeV)

        # argmax
        self.assertEqual(j.argmax(), 5)

        # min
        self.assertEqual(j.min(), 0 * q.MeV)

        # argmin
        self.assertEqual(m.argmin(), 0)

        # ptp
        self.assertEqual(m.ptp(), 8 * q.MeV)

        # clip
        self.numAssertEqual(
            j.clip(max=5*q.MeV),
            [1, 0, 5, 5, 0, 5] * q.MeV
        )
        self.numAssertEqual(
            j.clip(min=1*q.MeV),
            [1, 1, 5, 6, 1, 9] * q.MeV
        )
        self.numAssertEqual(
            j.clip(min=1*q.MeV, max=5*q.MeV),
            [1, 1, 5, 5, 1, 5] * q.MeV
        )
        self.assertRaises(ValueError, j.clip)
        self.assertRaises(ValueError, j.clip, 1)

        # round
        p = [1, 3.00001, 3, .6, 1000] * q.J
        self.numAssertEqual(p.round(0), [1, 3., 3, 1, 1000] * q.J)
        self.numAssertEqual(p.round(-1), [0, 0, 0, 0, 1000] * q.J)
        self.numAssertEqual(p.round(3), [1, 3., 3, .6, 1000] * q.J)

        # trace
        d = [[1., 2., 3., 4.], [1., 2., 3., 4.],[1., 2., 3., 4.]]*q.A
        self.numAssertEqual(d.trace(), (1+2+3) * q.A)

        # cumsum
        self.numAssertEqual(
            p.cumsum(),
            [1, 4.00001, 7.00001, 1 + 3.00001 + 3 + .6, 1007.60001] * q.J
        )

        # mean
        self.assertEqual(p.mean(), 201.520002 * q.J)

        # var
        self.assertAlmostEqual(
            p.var(),
            ((1 - 201.520002)**2 + (3.00001 -201.520002)**2 + \
                (3- 201.520002)**2 + (.6 - 201.520002) **2 + \
                (1000-201.520002)**2) / 5 * q.J**2
        )

        # std
        self.assertAlmostEqual(
            p.std(),
            np.sqrt(((1 - 201.520002)**2 + (3.00001 -201.520002)**2 + \
                (3- 201.520002) **2 + (.6 - 201.520002) **2 + \
                (1000-201.520002)**2) / 5) * q.J
            )

        # prod
        o = [1, 3, 2] * q.kPa
        self.assertEqual(o.prod(), 6 * q.kPa**3)

        # cumprod
        self.assertRaises(ValueError, o.cumprod)

        f = [1, 2, 3] * q.dimensionless
        self.numAssertEqual(f.cumprod(), [1,2,6] * q.dimensionless)

        #test conj and conjugate()
        w1 = [1 - 5j, 3 , 6 + 1j] * q.MeV
        self.numAssertEqual(
            w1.conj().magnitude,
            np.array([1-5j, 3, 6+1j]).conj()
        )
        self.numAssertEqual(
            w1.conjugate().magnitude,
            np.array([1-5j, 3, 6+1j]).conjugate()
        )


if __name__ == "__main__":
    run_module_suite()
