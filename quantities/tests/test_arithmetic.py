# -*- coding: utf-8 -*-

import operator as op
from functools import partial
import sys

import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase


if sys.version.startswith('3'):
    long = int


def rand(dtype, *args):
    if np.dtype(dtype).kind == 'c':
        return dtype(
            10*np.random.rand(*args)+10j*np.random.rand(*args),
        )
    else:
        return dtype(10*np.random.rand(*args))

def check(f, *args, **kwargs):
    new = partial(f, *args)
    new.__name__ = f.__name__
    new.__module__ = f.__module__

    try:
        new = kwargs['fails_if'](new)
    except KeyError:
        pass
    desc = [f.__name__]
    for arg in args:
        try:
            desc.append(arg[0].dtype.name)
        except AttributeError:
            desc.append(arg[0].__class__.__name__)
        except (IndexError, TypeError):
            try:
                desc.append(arg.dtype.name)
            except AttributeError:
                pass
        c = arg.__class__.__name__
        if c != desc[-1]:
            desc.append(c)

    new.description = '_'.join(desc)
    return (new, )


class iter_dtypes(object):

    def __init__(self):
        self._i = 1
        self._typeDict = np.typeDict.copy()
        self._typeDict[17] = int
        self._typeDict[18] = long
        self._typeDict[19] = float
        self._typeDict[20] = complex

    def __iter__(self):
        return self

    def __next__(self):
        if self._i > 20:
            raise StopIteration

        i = self._i
        self._i += 1
        return self._typeDict[i]

    def next(self):
        return self.__next__()

def get_dtypes():
    return list(iter_dtypes())


class iter_types(object):

    def __init__(self, dtype):
        self._index = -1
        self._dtype = dtype

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index > 2:
            raise StopIteration
        if self._index > 0 and self._dtype in (int, long, float, complex):
            raise StopIteration
        if self._index == 0:
            return rand(self._dtype)
        if self._index == 1:
            return rand(self._dtype, 5).tolist()
        if self._index == 2:
            return rand(self._dtype, 5)

    def next(self):
        return self.__next__()


class TestDTypes(TestCase):

    def check_mul(self, m1, m2):
        self.assertQuantityEqual(pq.m*m2, Quantity(m2, 'm'))

        q1 = Quantity(m1, 'm')
        q2 = Quantity(m2, 's')
        a1 = np.asarray(m1)
        a2 = np.asarray(m2)
        self.assertQuantityEqual(q1*m2, Quantity(a1*a2, 'm'))
        self.assertQuantityEqual(q1*q2, Quantity(a1*a2, 'm*s'))

    def check_rmul(self, m1, m2):
        self.assertQuantityEqual(m1*pq.m, Quantity(m1, 'm'))

        q2 = Quantity(m2, 's')
        a1 = np.asarray(m1)
        a2 = np.asarray(m2)
        self.assertQuantityEqual(m1*q2, Quantity(a1*a2, 's'))

    def test_mul(self):
        dtypes = get_dtypes()
        for i in get_dtypes()[::3]:
            for j in get_dtypes()[::3]:
                for x in iter_types(i):
                    for y in iter_types(j):
                        self.check_mul(x, y)
                        self.check_rmul(x, y)
            dtypes.pop(0)

    def test_mixed_addition(self):
        self.assertQuantityEqual(1*pq.ft + 1*pq.m, 4.280839895 * pq.ft)
        self.assertQuantityEqual(1*pq.ft + pq.m, 4.280839895 * pq.ft)
        self.assertQuantityEqual(pq.ft + 1*pq.m, 4.280839895 * pq.ft)
        self.assertQuantityEqual(pq.ft + pq.m, 4.280839895 * pq.ft)
        self.assertQuantityEqual(op.iadd(1*pq.ft, 1*pq.m), 4.280839895 * pq.ft)
        self.assertRaises(ValueError, lambda: 10*pq.J + 3*pq.m)
        self.assertRaises(ValueError, lambda: op.iadd(10*pq.J, 3*pq.m))

    def test_mod(self):
        self.assertQuantityEqual(10*pq.m % (3*pq.m), 1*pq.m)
        self.assertQuantityEqual(
            10*pq.m % (3*pq.m).rescale('ft'),
            10*pq.m % (3*pq.m)
        )
        self.assertRaises(ValueError, lambda: 10*pq.J % (3*pq.m))

    def test_imod(self):
        x = 10*pq.m
        x %= 3*pq.m
        self.assertQuantityEqual(x, 1*pq.m)

        x = 10*pq.m
        x %= (3*pq.m).rescale('ft')
        self.assertQuantityEqual(x, 10*pq.m % (3*pq.m))

        self.assertRaises(ValueError, lambda: op.imod(10*pq.J, 3*pq.m))

    def test_fmod(self):
        self.assertQuantityEqual(np.fmod(10*pq.m, (3*pq.m)), 1*pq.m)
        self.assertRaises(ValueError, np.fmod, 10*pq.J, 3*pq.m)

    def test_remainder(self):
        self.assertQuantityEqual(np.remainder(10*pq.m, (3*pq.m)), 1*pq.m)
        self.assertRaises(ValueError, np.remainder, 10*pq.J, 3*pq.m)

    def test_negative(self):
        self.assertQuantityEqual(
            -pq.m,
            Quantity(-1, 'm')
        )
        self.assertQuantityEqual(
            -Quantity(5, 'm'),
            Quantity(-5, 'm')
        )
        self.assertQuantityEqual(
            -Quantity(-5.0, 'm'),
            Quantity(5.0, 'm')
        )

    def test_addition(self):
        self.assertQuantityEqual(
            pq.eV + pq.eV,
            2*pq.eV
        )
        self.assertQuantityEqual(
            pq.eV + -pq.eV,
            0*pq.eV
        )
        self.assertQuantityEqual(
            pq.eV + 5*pq.eV,
            6*pq.eV
        )
        self.assertQuantityEqual(
            5*pq.eV + pq.eV,
            6*pq.eV
        )
        self.assertQuantityEqual(
            5*pq.eV + 6*pq.eV,
            11*pq.eV
        )
        self.assertQuantityEqual(
            pq.rem + [1, 2, 3]*pq.rem,
            [2, 3, 4]*pq.rem
        )
        self.assertQuantityEqual(
            [1, 2, 3]*pq.rem + pq.rem,
            [2, 3, 4]*pq.rem
        )
        self.assertQuantityEqual(
            5*pq.rem + [1, 2, 3]*pq.rem,
            [6, 7, 8]*pq.rem
        )
        self.assertQuantityEqual(
            [1, 2, 3]*pq.rem + 5*pq.rem,
            [6, 7, 8]*pq.rem
        )
        self.assertQuantityEqual(
            [1, 2, 3]*pq.hp + [1, 2, 3]*pq.hp,
            [2, 4, 6]*pq.hp
        )

        self.assertRaises(ValueError, op.add, pq.kPa, pq.lb)
        self.assertRaises(ValueError, op.add, pq.kPa, 10)
        self.assertRaises(ValueError, op.add, 1*pq.kPa, 5*pq.lb)
        self.assertRaises(ValueError, op.add, 1*pq.kPa, pq.lb)
        self.assertRaises(ValueError, op.add, 1*pq.kPa, 5)
        self.assertRaises(ValueError, op.add, [1, 2, 3]*pq.kPa, [1, 2, 3]*pq.lb)
        self.assertRaises(ValueError, op.add, [1, 2, 3]*pq.kPa, 5*pq.lb)
        self.assertRaises(ValueError, op.add, [1, 2, 3]*pq.kPa, pq.lb)
        self.assertRaises(ValueError, op.add, [1, 2, 3]*pq.kPa, 5)

    def test_in_place_addition(self):
        x = 1*pq.m
        x += pq.m
        self.assertQuantityEqual(x, pq.m+pq.m)

        x = 1*pq.m
        x += -pq.m
        self.assertQuantityEqual(x, 0*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x += pq.m
        self.assertQuantityEqual(x, [2, 3, 4, 5]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x += x
        self.assertQuantityEqual(x, [2, 4, 6, 8]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[:2] += pq.m
        self.assertQuantityEqual(x, [2, 3, 3, 4]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[:2] += -pq.m
        self.assertQuantityEqual(x, [0, 1, 3, 4]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[:2] += [1, 2]*pq.m
        self.assertQuantityEqual(x, [2, 4, 3, 4]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[::2] += [1, 2]*pq.m
        self.assertQuantityEqual(x, [2, 2, 5, 4]*pq.m)

        self.assertRaises(ValueError, op.iadd, 1*pq.m, 1)
        self.assertRaises(ValueError, op.iadd, 1*pq.m, pq.J)
        self.assertRaises(ValueError, op.iadd, 1*pq.m, 5*pq.J)
        self.assertRaises(ValueError, op.iadd, [1, 2, 3]*pq.m, 1)
        self.assertRaises(ValueError, op.iadd, [1, 2, 3]*pq.m, pq.J)
        self.assertRaises(ValueError, op.iadd, [1, 2, 3]*pq.m, 5*pq.J)

    def test_subtraction(self):
        self.assertQuantityEqual(
            pq.eV - pq.eV,
            0*pq.eV
        )
        self.assertQuantityEqual(
            5*pq.eV - pq.eV,
            4*pq.eV
        )
        self.assertQuantityEqual(
            pq.eV - 4*pq.eV,
            -3*pq.eV
        )
        self.assertQuantityEqual(
            pq.rem - [1, 2, 3]*pq.rem,
            [0, -1, -2]*pq.rem
        )
        self.assertQuantityEqual(
            [1, 2, 3]*pq.rem - pq.rem,
            [0, 1, 2]*pq.rem
        )
        self.assertQuantityEqual(
            5*pq.rem - [1, 2, 3]*pq.rem,
            [4, 3, 2]*pq.rem
        )
        self.assertQuantityEqual(
            [1, 2, 3]*pq.rem - 5*pq.rem,
            [-4, -3, -2]*pq.rem
        )
        self.assertQuantityEqual(
            [3, 3, 3]*pq.hp - [1, 2, 3]*pq.hp,
            [2, 1, 0]*pq.hp
        )

        self.assertRaises(ValueError, op.sub, pq.kPa, pq.lb)
        self.assertRaises(ValueError, op.sub, pq.kPa, 10)
        self.assertRaises(ValueError, op.sub, 1*pq.kPa, 5*pq.lb)
        self.assertRaises(ValueError, op.sub, 1*pq.kPa, pq.lb)
        self.assertRaises(ValueError, op.sub, 1*pq.kPa, 5)
        self.assertRaises(ValueError, op.sub, [1, 2, 3]*pq.kPa, [1, 2, 3]*pq.lb)
        self.assertRaises(ValueError, op.sub, [1, 2, 3]*pq.kPa, 5*pq.lb)
        self.assertRaises(ValueError, op.sub, [1, 2, 3]*pq.kPa, pq.lb)
        self.assertRaises(ValueError, op.sub, [1, 2, 3]*pq.kPa, 5)

    def test_in_place_subtraction(self):
        x = 1*pq.m
        x -= pq.m
        self.assertQuantityEqual(x, 0*pq.m)

        x = 1*pq.m
        x -= -pq.m
        self.assertQuantityEqual(x, 2*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x -= pq.m
        self.assertQuantityEqual(x, [0, 1, 2, 3]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x -= [1, 1, 1, 1]*pq.m
        self.assertQuantityEqual(x, [0, 1, 2, 3]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[:2] -= pq.m
        self.assertQuantityEqual(x, [0, 1, 3, 4]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[:2] -= -pq.m
        self.assertQuantityEqual(x, [2, 3, 3, 4]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[:2] -= [1, 2]*pq.m
        self.assertQuantityEqual(x, [0, 0, 3, 4]*pq.m)

        x = [1, 2, 3, 4]*pq.m
        x[::2] -= [1, 2]*pq.m
        self.assertQuantityEqual(x, [0, 2, 1, 4]*pq.m)

        self.assertRaises(ValueError, op.isub, 1*pq.m, 1)
        self.assertRaises(ValueError, op.isub, 1*pq.m, pq.J)
        self.assertRaises(ValueError, op.isub, 1*pq.m, 5*pq.J)
        self.assertRaises(ValueError, op.isub, [1, 2, 3]*pq.m, 1)
        self.assertRaises(ValueError, op.isub, [1, 2, 3]*pq.m, pq.J)
        self.assertRaises(ValueError, op.isub, [1, 2, 3]*pq.m, 5*pq.J)

    def test_powering(self):
        # test raising a quantity to a power
        self.assertQuantityEqual((5.5 * pq.cm)**5, (5.5**5) * (pq.cm**5))
        self.assertQuantityEqual((5.5 * pq.cm)**0, (5.5**0) * pq.dimensionless)

        # must also work with compound units
        self.assertQuantityEqual((5.5 * pq.J)**5, (5.5**5) * (pq.J**5))

        # does powering work with arrays?
        self.assertQuantityEqual(
            (np.array([1, 2, 3, 4, 5]) * pq.kg)**3,
            np.array([1, 8, 27, 64, 125]) * pq.kg**3
        )

        def q_pow_r(q1, q2):
            return q1 ** q2

        self.assertRaises(ValueError, q_pow_r, 10.0 * pq.m, 10 * pq.J)
        self.assertRaises(ValueError, q_pow_r, 10.0 * pq.m, np.array([1, 2, 3]))

        self.assertQuantityEqual( (10 * pq.J) ** (2 * pq.J/pq.J) , 100 * pq.J**2 )

        # test rpow here
        self.assertRaises(ValueError, q_pow_r, 10.0, 10 * pq.J)

        self.assertQuantityEqual(10**(2*pq.J/pq.J), 100)

        def ipow(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, ipow, 1*pq.m, [1, 2])
