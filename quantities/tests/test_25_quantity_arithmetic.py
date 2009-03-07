# -*- coding: utf-8 -*-

import operator as op
from functools import partial

from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq
from quantities.utilities import (
    assert_quantity_equal, assert_quantity_almost_equal
)
# TODO: remove this mask once bug numpy bugs 826 and 1026 are fixed
assert_quantity_equal = assert_array_equal

def rand(dtype, *args):
    try:
        return dtype(10*np.random.rand(*args)+10j*np.random.rand(*args))
    except TypeError:
        return dtype(10*np.random.rand(*args))

def check(f, v1, v2, name, dt1, dt2):
    f = partial(f, v1, v2)
    f.description = '_'.join([name, dt1.__name__, dt2.__name__])
    return (f, )


class iter_dtypes(object):

    def __init__(self):
        self._i = 1

    def __iter__(self):
        return self

    def next(self):
        if self._i > 16:
            raise StopIteration

        i = self._i
        self._i += 1
        return np.typeDict[i]


def check_mul(m1, m2):
    assert_quantity_equal(m1*pq.m, pq.Quantity(m1, 'm'))
    assert_quantity_equal(pq.m*m1, pq.Quantity(m1, 'm'))
    assert_quantity_equal(m2*pq.m, pq.Quantity(m2, 'm'))
    assert_quantity_equal(pq.m*m2, pq.Quantity(m2, 'm'))

    q1 = pq.Quantity(m1, 'm')
    q2 = pq.Quantity(m2, 's')
    a1 = np.asarray(m1)
    a2 = np.asarray(m2)
    assert_quantity_equal(q1*m2, pq.Quantity(a1*a2, 'm'))
    assert_quantity_equal(m2*q1, pq.Quantity(a2*a1, 'm'))
    assert_quantity_equal(m1*q2, pq.Quantity(a1*a2, 's'))
    assert_quantity_equal(q2*m1, pq.Quantity(a2*a1, 's'))
    assert_quantity_equal(q1*q2, pq.Quantity(a1*a2, 'm*s'))
    assert_quantity_equal(q2*q1, pq.Quantity(a2*a1, 'm*s'))

def test_mul():
    for i in iter_dtypes():
        for j in iter_dtypes():
            yield check(
                check_mul, rand(i), rand(j), 'test_ss_mul', i, j
            )
            yield check(
                check_mul, rand(i), list(rand(j, 5)), 'test_ss_mul', i, j
            )
            yield check(
                check_mul, rand(i), rand(j, 5), 'test_sa_mul', i, j
            )
            yield check(
                check_mul, list(rand(i, 5)), list(rand(i, 5)),
                'test_ii_mul', i, j
            )
            yield check(
                check_mul, list(rand(i, 5)), rand(j, 5), 'test_ia_mul', i, j
            )
            yield check(
                check_mul, rand(i, 5), rand(j, 5), 'test_aa_mul', i, j
            )




#def test_negative():
#    assert_array_equal(
#        -pq.m,
#        pq.Quantity(-1, 'm')
#    )
#    assert_array_equal(
#        -pq.Quantity(5, 'm'),
#        pq.Quantity(-5, 'm')
#    )
#    assert_array_equal(
#        -pq.Quantity(-5.0, 'm'),
#        pq.Quantity(-5.0, 'm')
#    )
#    assert_array_equal(
#        -[]pq.m,
#        pq.Quantity(-1, 'm')
#    )


#def test_multiplication():
#    assert_array_equal(
#        2 * pq.eV,
#        2*pq.eV
#    )
#    assert_array_equal(
#        2 * -pq.eV,
#        -2*pq.eV
#    )
#    assert_array_equal(
#        2 * 5*pq.eV,
#        10*pq.eV
#    )
#    assert_array_equal(
#        5*pq.eV * 2,
#        10*pq.eV
#    )
#    assert_array_equal(
#        2 * ([1, 2, 3]*pq.rem),
#        [2, 4, 6]*pq.rem
#    )
#    assert_array_equal(
#        [1, 2, 3]*pq.rem * 2,
#        [2, 4, 6]*pq.rem
#    )
#    assert_array_equal(
#        [1, 2, 3] * ([1, 2, 3]*pq.hp),
#        [1, 4, 9]*pq.hp
#    )
#
##    assert_array_equal(
##        [1, 2, 3]*pq.m * [1, 2, 3]*pq.hp,
##        pq.Quantity([1, 4, 9], 'm*hp')
##    )
##    assert_array_equal(
##        [1, 2, 3]*pq.m * [1, 2, 3]*pq.m,
##        pq.Quantity([1, 4, 9], 'm**2')
##    )
##    assert_array_equal(
##        [1, 2, 3]*pq.m * [1, 2, 3]*pq.m**-1,
##        pq.Quantity([1, 4, 9], '')
##    )
#
## this returns an array instead of a quantity:
##def test_mul_with_list():
##    assert_array_equal(
##        pq.m * [1, 2, 3],
##        pq.Quantity([1, 4, 9], 'm*hp')
##    )
#
##def test_in_place_multiplication():
##    x = 1*pq.m
##    x += pq.m
##    assert_array_equal(x, pq.m+pq.m)
##
##    x = 1*pq.m
##    x += -pq.m
##    assert_array_equal(x, 0*pq.m)
##
##    x = [1, 2, 3, 4]*pq.m
##    x += pq.m
##    assert_array_equal(x, [2, 3, 4, 5]*pq.m)
##
##    x = [1, 2, 3, 4]*pq.m
##    x += x
##    assert_array_equal(x, [2, 4, 6, 8]*pq.m)
##
##    x = [1, 2, 3, 4]*pq.m
##    x[:2] += pq.m
##    assert_array_equal(x, [2, 3, 3, 4]*pq.m)
##
##    x = [1, 2, 3, 4]*pq.m
##    x[:2] += -pq.m
##    assert_array_equal(x, [0, 1, 3, 4]*pq.m)
##
##    x = [1, 2, 3, 4]*pq.m
##    x[:2] += [1, 2]*pq.m
##    assert_array_equal(x, [2, 4, 3, 4]*pq.m)
##
##    x = [1, 2, 3, 4]*pq.m
##    x[::2] += [1, 2]*pq.m
##    assert_array_equal(x, [2, 2, 5, 4]*pq.m)
##
##    assert_raises(ValueError, op.iadd, 1*pq.m, 1)
##    assert_raises(ValueError, op.iadd, 1*pq.m, pq.J)
##    assert_raises(ValueError, op.iadd, 1*pq.m, 5*pq.J)
##    assert_raises(ValueError, op.iadd, [1, 2, 3]*pq.m, 1)
##    assert_raises(ValueError, op.iadd, [1, 2, 3]*pq.m, pq.J)
##    assert_raises(ValueError, op.iadd, [1, 2, 3]*pq.m, 5*pq.J)
#
#def test_addition():
#    assert_array_equal(
#        pq.eV + pq.eV,
#        2*pq.eV
#    )
#    assert_array_equal(
#        pq.eV + -pq.eV,
#        0*pq.eV
#    )
#    assert_array_equal(
#        pq.eV + 5*pq.eV,
#        6*pq.eV
#    )
#    assert_array_equal(
#        5*pq.eV + pq.eV,
#        6*pq.eV
#    )
#    assert_array_equal(
#        5*pq.eV + 6*pq.eV,
#        11*pq.eV
#    )
#    assert_array_equal(
#        pq.rem + [1, 2, 3]*pq.rem,
#        [2, 3, 4]*pq.rem
#    )
#    assert_array_equal(
#        [1, 2, 3]*pq.rem + pq.rem,
#        [2, 3, 4]*pq.rem
#    )
#    assert_array_equal(
#        5*pq.rem + [1, 2, 3]*pq.rem,
#        [6, 7, 8]*pq.rem
#    )
#    assert_array_equal(
#        [1, 2, 3]*pq.rem + 5*pq.rem,
#        [6, 7, 8]*pq.rem
#    )
#    assert_array_equal(
#        [1, 2, 3]*pq.hp + [1, 2, 3]*pq.hp,
#        [2, 4, 6]*pq.hp
#    )
#
#    assert_raises(ValueError, op.add, pq.kPa, pq.lb)
#    assert_raises(ValueError, op.add, pq.kPa, 10)
#    assert_raises(ValueError, op.add, 1*pq.kPa, 5*pq.lb)
#    assert_raises(ValueError, op.add, 1*pq.kPa, pq.lb)
#    assert_raises(ValueError, op.add, 1*pq.kPa, 5)
#    assert_raises(ValueError, op.add, [1, 2, 3]*pq.kPa, [1, 2, 3]*pq.lb)
#    assert_raises(ValueError, op.add, [1, 2, 3]*pq.kPa, 5*pq.lb)
#    assert_raises(ValueError, op.add, [1, 2, 3]*pq.kPa, pq.lb)
#    assert_raises(ValueError, op.add, [1, 2, 3]*pq.kPa, 5)
#
#def test_in_place_addition():
#    x = 1*pq.m
#    x += pq.m
#    assert_array_equal(x, pq.m+pq.m)
#
#    x = 1*pq.m
#    x += -pq.m
#    assert_array_equal(x, 0*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x += pq.m
#    assert_array_equal(x, [2, 3, 4, 5]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x += x
#    assert_array_equal(x, [2, 4, 6, 8]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[:2] += pq.m
#    assert_array_equal(x, [2, 3, 3, 4]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[:2] += -pq.m
#    assert_array_equal(x, [0, 1, 3, 4]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[:2] += [1, 2]*pq.m
#    assert_array_equal(x, [2, 4, 3, 4]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[::2] += [1, 2]*pq.m
#    assert_array_equal(x, [2, 2, 5, 4]*pq.m)
#
#    assert_raises(ValueError, op.iadd, 1*pq.m, 1)
#    assert_raises(ValueError, op.iadd, 1*pq.m, pq.J)
#    assert_raises(ValueError, op.iadd, 1*pq.m, 5*pq.J)
#    assert_raises(ValueError, op.iadd, [1, 2, 3]*pq.m, 1)
#    assert_raises(ValueError, op.iadd, [1, 2, 3]*pq.m, pq.J)
#    assert_raises(ValueError, op.iadd, [1, 2, 3]*pq.m, 5*pq.J)
#
#def test_subtraction():
#    assert_array_equal(
#        pq.eV - pq.eV,
#        0*pq.eV
#    )
#    assert_array_equal(
#        5*pq.eV - pq.eV,
#        4*pq.eV
#    )
#    assert_array_equal(
#        pq.eV - 4*pq.eV,
#        -3*pq.eV
#    )
#    assert_array_equal(
#        pq.rem - [1, 2, 3]*pq.rem,
#        [0, -1, -2]*pq.rem
#    )
#    assert_array_equal(
#        [1, 2, 3]*pq.rem - pq.rem,
#        [0, 1, 2]*pq.rem
#    )
#    assert_array_equal(
#        5*pq.rem - [1, 2, 3]*pq.rem,
#        [4, 3, 2]*pq.rem
#    )
#    assert_array_equal(
#        [1, 2, 3]*pq.rem - 5*pq.rem,
#        [-4, -3, -2]*pq.rem
#    )
#    assert_array_equal(
#        [3, 3, 3]*pq.hp - [1, 2, 3]*pq.hp,
#        [2, 1, 0]*pq.hp
#    )
#
#    assert_raises(ValueError, op.sub, pq.kPa, pq.lb)
#    assert_raises(ValueError, op.sub, pq.kPa, 10)
#    assert_raises(ValueError, op.sub, 1*pq.kPa, 5*pq.lb)
#    assert_raises(ValueError, op.sub, 1*pq.kPa, pq.lb)
#    assert_raises(ValueError, op.sub, 1*pq.kPa, 5)
#    assert_raises(ValueError, op.sub, [1, 2, 3]*pq.kPa, [1, 2, 3]*pq.lb)
#    assert_raises(ValueError, op.sub, [1, 2, 3]*pq.kPa, 5*pq.lb)
#    assert_raises(ValueError, op.sub, [1, 2, 3]*pq.kPa, pq.lb)
#    assert_raises(ValueError, op.sub, [1, 2, 3]*pq.kPa, 5)
#
#def test_in_place_subtraction():
#    x = 1*pq.m
#    x -= pq.m
#    assert_array_equal(x, 0*pq.m)
#
#    x = 1*pq.m
#    x -= -pq.m
#    assert_array_equal(x, 2*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x -= pq.m
#    assert_array_equal(x, [0, 1, 2, 3]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x -= [1, 1, 1, 1]*pq.m
#    assert_array_equal(x, [0, 1, 2, 3]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[:2] -= pq.m
#    assert_array_equal(x, [0, 1, 3, 4]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[:2] -= -pq.m
#    assert_array_equal(x, [2, 3, 3, 4]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[:2] -= [1, 2]*pq.m
#    assert_array_equal(x, [0, 0, 3, 4]*pq.m)
#
#    x = [1, 2, 3, 4]*pq.m
#    x[::2] -= [1, 2]*pq.m
#    assert_array_equal(x, [0, 2, 1, 4]*pq.m)
#
#    assert_raises(ValueError, op.isub, 1*pq.m, 1)
#    assert_raises(ValueError, op.isub, 1*pq.m, pq.J)
#    assert_raises(ValueError, op.isub, 1*pq.m, 5*pq.J)
#    assert_raises(ValueError, op.isub, [1, 2, 3]*pq.m, 1)
#    assert_raises(ValueError, op.isub, [1, 2, 3]*pq.m, pq.J)
#    assert_raises(ValueError, op.isub, [1, 2, 3]*pq.m, 5*pq.J)
#
#def test_powering():
#    # test raising a quantity to a power
#    assert_array_almost_equal((5.5 * pq.cm)**5, (5.5**5) * (pq.cm**5))
#    assert_array_equal(str((5.5 * pq.cm)**5), str((5.5**5) * (pq.cm**5)))
#
#    # must also work with compound units
#    assert_array_almost_equal((5.5 * pq.J)**5, (5.5**5) * (pq.J**5))
#    assert_array_equal(str((5.5 * pq.J)**5), str((5.5**5) * (pq.J**5)))
#
#    # does powering work with arrays?
#    temp = np.array([1, 2, 3, 4, 5]) * pq.kg
#    temp2 = (np.array([1, 8, 27, 64, 125]) **2) * pq.kg**6
#
#    assert_array_equal(
#        str(temp**3),
#        "[   1.    8.   27.   64.  125.] kg³"
#    )
#    assert_array_equal(str(temp**6), str(temp2))
#
#    def q_pow_r(q1, q2):
#        return q1 ** q2
#
#    assert_raises(ValueError, q_pow_r, 10.0 * pq.m, 10 * pq.J)
#    assert_raises(ValueError, q_pow_r, 10.0 * pq.m, np.array([1, 2, 3]))
#
#    assert_array_equal( (10 * pq.J) ** (2 * pq.J/pq.J) , 100 * pq.J**2 )
#
#    # test rpow here
#    assert_raises(ValueError, q_pow_r, 10.0, 10 * pq.J)
#
#    assert_array_equal(10**(2*pq.J/pq.J), 100)
#
#    # in-place
##        temp1 = 1*m
##        temp2 = 1*m
##        temp1 **= 2
##        assert_array_equal(str(temp1), str(temp2*temp2))
##
##        temp1 = [1, 2, 3, 4]*m
##        temp2 = [1, 2, 3, 4]*m
##        temp1 **= 2
##        assert_array_equal(str(temp1), str(temp2*temp2))
#
#    def ipow(q1, q2):
#        q1 -= q2
#    assert_raises(ValueError, ipow, 1*pq.m, [1, 2])


if __name__ == "__main__":
    run_module_suite()
