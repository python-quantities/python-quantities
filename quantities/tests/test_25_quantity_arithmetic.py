# -*- coding: utf-8 -*-

import operator as op
from functools import partial

from numpy.testing import *
from numpy.testing import *
from numpy.testing.decorators import knownfailureif as fails_if
from numpy.testing.decorators import skipif as skip_if

import numpy as np
from .. import units
from ..quantity import Quantity

from . import assert_quantity_equal, assert_quantity_almost_equal


def rand(dtype, *args):
    if dtype in (np.complex64, np.complex128, np.complex256):
        return dtype(
            10*np.random.rand(*args)+10j*np.random.rand(*args),
        )
    else:
        return dtype(10*np.random.rand(*args))

def check(f, *args, **kwargs):
    new = partial(f, *args)
    new.__name__ = f.__name__
    new.__module__ = f.__module__
    new.func_code = f.func_code

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

    def next(self):
        if self._i > 20:
            raise StopIteration

        i = self._i
        self._i += 1
        return self._typeDict[i]

def get_dtypes():
    return list(iter_dtypes())


class iter_types(object):

    def __init__(self, dtype):
        self._index = -1
        self._dtype = dtype

    def __iter__(self):
        return self

    def next(self):
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

def check_mul(m1, m2):
    assert_quantity_equal(units.m*m2, Quantity(m2, 'm'))

    q1 = Quantity(m1, 'm')
    q2 = Quantity(m2, 's')
    a1 = np.asarray(m1)
    a2 = np.asarray(m2)
    assert_quantity_equal(q1*m2, Quantity(a1*a2, 'm'))
    assert_quantity_equal(q1*q2, Quantity(a1*a2, 'm*s'))

def check_rmul(m1, m2):
    assert_quantity_equal(m1*units.m, Quantity(m1, 'm'))

    q2 = Quantity(m2, 's')
    a1 = np.asarray(m1)
    a2 = np.asarray(m2)
    assert_quantity_equal(m1*q2, Quantity(a1*a2, 's'))

def test_mul():
    dtypes = get_dtypes()
    while dtypes:
        i = dtypes[0]
        for j in dtypes:
            for x in iter_types(i):
                for y in iter_types(j):
                    yield check(check_mul, x, y)
                    yield check(check_rmul, x, y)
        dtypes.pop(0)

def test_mixed_addition():
    assert_quantity_almost_equal(1*units.ft + 1*units.m, 4.280839895 * units.ft)
    assert_quantity_almost_equal(1*units.ft + units.m, 4.280839895 * units.ft)
    assert_quantity_almost_equal(units.ft + 1*units.m, 4.280839895 * units.ft)
    assert_quantity_almost_equal(units.ft + units.m, 4.280839895 * units.ft)
    assert_quantity_almost_equal(op.iadd(1*units.ft, 1*units.m), 4.280839895 * units.ft)
    assert_raises(ValueError, lambda: 10*units.J + 3*units.m)
    assert_raises(ValueError, lambda: op.iadd(10*units.J, 3*units.m))

def test_mod():
    assert_quantity_almost_equal(10*units.m % (3*units.m), 1*units.m)
    assert_quantity_almost_equal(
        10*units.m % (3*units.m).rescale('ft'),
        10*units.m % (3*units.m)
    )
    assert_raises(ValueError, lambda: 10*units.J % (3*units.m))

def test_imod():
    x = 10*units.m
    x %= 3*units.m
    assert_quantity_almost_equal(x, 1*units.m)

    x = 10*units.m
    x %= (3*units.m).rescale('ft')
    assert_quantity_almost_equal(x, 10*units.m % (3*units.m))

    assert_raises(ValueError, lambda: op.imod(10*units.J, 3*units.m))

def test_fmod():
    assert_quantity_almost_equal(np.fmod(10*units.m, (3*units.m)), 1*units.m)
    assert_raises(ValueError, np.fmod, 10*units.J, 3*units.m)

def test_remainder():
    assert_quantity_almost_equal(np.remainder(10*units.m, (3*units.m)), 1*units.m)
    assert_raises(ValueError, np.remainder, 10*units.J, 3*units.m)

def test_negative():
    assert_quantity_equal(
        -units.m,
        Quantity(-1, 'm')
    )
    assert_quantity_equal(
        -Quantity(5, 'm'),
        Quantity(-5, 'm')
    )
    assert_quantity_equal(
        -Quantity(-5.0, 'm'),
        Quantity(5.0, 'm')
    )

def test_addition():
    assert_quantity_equal(
        units.eV + units.eV,
        2*units.eV
    )
    assert_quantity_equal(
        units.eV + -units.eV,
        0*units.eV
    )
    assert_quantity_equal(
        units.eV + 5*units.eV,
        6*units.eV
    )
    assert_quantity_equal(
        5*units.eV + units.eV,
        6*units.eV
    )
    assert_quantity_equal(
        5*units.eV + 6*units.eV,
        11*units.eV
    )
    assert_quantity_equal(
        units.rem + [1, 2, 3]*units.rem,
        [2, 3, 4]*units.rem
    )
    assert_quantity_equal(
        [1, 2, 3]*units.rem + units.rem,
        [2, 3, 4]*units.rem
    )
    assert_quantity_equal(
        5*units.rem + [1, 2, 3]*units.rem,
        [6, 7, 8]*units.rem
    )
    assert_quantity_equal(
        [1, 2, 3]*units.rem + 5*units.rem,
        [6, 7, 8]*units.rem
    )
    assert_quantity_equal(
        [1, 2, 3]*units.hp + [1, 2, 3]*units.hp,
        [2, 4, 6]*units.hp
    )

    assert_raises(ValueError, op.add, units.kPa, units.lb)
    assert_raises(ValueError, op.add, units.kPa, 10)
    assert_raises(ValueError, op.add, 1*units.kPa, 5*units.lb)
    assert_raises(ValueError, op.add, 1*units.kPa, units.lb)
    assert_raises(ValueError, op.add, 1*units.kPa, 5)
    assert_raises(ValueError, op.add, [1, 2, 3]*units.kPa, [1, 2, 3]*units.lb)
    assert_raises(ValueError, op.add, [1, 2, 3]*units.kPa, 5*units.lb)
    assert_raises(ValueError, op.add, [1, 2, 3]*units.kPa, units.lb)
    assert_raises(ValueError, op.add, [1, 2, 3]*units.kPa, 5)

def test_in_place_addition():
    x = 1*units.m
    x += units.m
    assert_quantity_equal(x, units.m+units.m)

    x = 1*units.m
    x += -units.m
    assert_quantity_equal(x, 0*units.m)

    x = [1, 2, 3, 4]*units.m
    x += units.m
    assert_quantity_equal(x, [2, 3, 4, 5]*units.m)

    x = [1, 2, 3, 4]*units.m
    x += x
    assert_quantity_equal(x, [2, 4, 6, 8]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[:2] += units.m
    assert_quantity_equal(x, [2, 3, 3, 4]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[:2] += -units.m
    assert_quantity_equal(x, [0, 1, 3, 4]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[:2] += [1, 2]*units.m
    assert_quantity_equal(x, [2, 4, 3, 4]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[::2] += [1, 2]*units.m
    assert_quantity_equal(x, [2, 2, 5, 4]*units.m)

    assert_raises(ValueError, op.iadd, 1*units.m, 1)
    assert_raises(ValueError, op.iadd, 1*units.m, units.J)
    assert_raises(ValueError, op.iadd, 1*units.m, 5*units.J)
    assert_raises(ValueError, op.iadd, [1, 2, 3]*units.m, 1)
    assert_raises(ValueError, op.iadd, [1, 2, 3]*units.m, units.J)
    assert_raises(ValueError, op.iadd, [1, 2, 3]*units.m, 5*units.J)

def test_subtraction():
    assert_quantity_equal(
        units.eV - units.eV,
        0*units.eV
    )
    assert_quantity_equal(
        5*units.eV - units.eV,
        4*units.eV
    )
    assert_quantity_equal(
        units.eV - 4*units.eV,
        -3*units.eV
    )
    assert_quantity_equal(
        units.rem - [1, 2, 3]*units.rem,
        [0, -1, -2]*units.rem
    )
    assert_quantity_equal(
        [1, 2, 3]*units.rem - units.rem,
        [0, 1, 2]*units.rem
    )
    assert_quantity_equal(
        5*units.rem - [1, 2, 3]*units.rem,
        [4, 3, 2]*units.rem
    )
    assert_quantity_equal(
        [1, 2, 3]*units.rem - 5*units.rem,
        [-4, -3, -2]*units.rem
    )
    assert_quantity_equal(
        [3, 3, 3]*units.hp - [1, 2, 3]*units.hp,
        [2, 1, 0]*units.hp
    )

    assert_raises(ValueError, op.sub, units.kPa, units.lb)
    assert_raises(ValueError, op.sub, units.kPa, 10)
    assert_raises(ValueError, op.sub, 1*units.kPa, 5*units.lb)
    assert_raises(ValueError, op.sub, 1*units.kPa, units.lb)
    assert_raises(ValueError, op.sub, 1*units.kPa, 5)
    assert_raises(ValueError, op.sub, [1, 2, 3]*units.kPa, [1, 2, 3]*units.lb)
    assert_raises(ValueError, op.sub, [1, 2, 3]*units.kPa, 5*units.lb)
    assert_raises(ValueError, op.sub, [1, 2, 3]*units.kPa, units.lb)
    assert_raises(ValueError, op.sub, [1, 2, 3]*units.kPa, 5)

def test_in_place_subtraction():
    x = 1*units.m
    x -= units.m
    assert_quantity_equal(x, 0*units.m)

    x = 1*units.m
    x -= -units.m
    assert_quantity_equal(x, 2*units.m)

    x = [1, 2, 3, 4]*units.m
    x -= units.m
    assert_quantity_equal(x, [0, 1, 2, 3]*units.m)

    x = [1, 2, 3, 4]*units.m
    x -= [1, 1, 1, 1]*units.m
    assert_quantity_equal(x, [0, 1, 2, 3]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[:2] -= units.m
    assert_quantity_equal(x, [0, 1, 3, 4]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[:2] -= -units.m
    assert_quantity_equal(x, [2, 3, 3, 4]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[:2] -= [1, 2]*units.m
    assert_quantity_equal(x, [0, 0, 3, 4]*units.m)

    x = [1, 2, 3, 4]*units.m
    x[::2] -= [1, 2]*units.m
    assert_quantity_equal(x, [0, 2, 1, 4]*units.m)

    assert_raises(ValueError, op.isub, 1*units.m, 1)
    assert_raises(ValueError, op.isub, 1*units.m, units.J)
    assert_raises(ValueError, op.isub, 1*units.m, 5*units.J)
    assert_raises(ValueError, op.isub, [1, 2, 3]*units.m, 1)
    assert_raises(ValueError, op.isub, [1, 2, 3]*units.m, units.J)
    assert_raises(ValueError, op.isub, [1, 2, 3]*units.m, 5*units.J)

def test_powering():
    # test raising a quantity to a power
    assert_quantity_almost_equal((5.5 * units.cm)**5, (5.5**5) * (units.cm**5))

    # must also work with compound units
    assert_quantity_almost_equal((5.5 * units.J)**5, (5.5**5) * (units.J**5))

    # does powering work with arrays?
    assert_quantity_equal(
        (np.array([1, 2, 3, 4, 5]) * units.kg)**3,
        np.array([1, 8, 27, 64, 125]) * units.kg**3
    )

    def q_pow_r(q1, q2):
        return q1 ** q2

    assert_raises(ValueError, q_pow_r, 10.0 * units.m, 10 * units.J)
    assert_raises(ValueError, q_pow_r, 10.0 * units.m, np.array([1, 2, 3]))

    assert_quantity_equal( (10 * units.J) ** (2 * units.J/units.J) , 100 * units.J**2 )

    # test rpow here
    assert_raises(ValueError, q_pow_r, 10.0, 10 * units.J)

    assert_quantity_equal(10**(2*units.J/units.J), 100)

    def ipow(q1, q2):
        q1 -= q2
    assert_raises(ValueError, ipow, 1*units.m, [1, 2])
