# -*- coding: utf-8 -*-

import operator as op

from .. import units as pq
from ..dimensionality import Dimensionality
from .common import TestCase

meter = Dimensionality({pq.m: 1})
meter_str = 'm'
centimeter = Dimensionality({pq.cm: 1})
centimeter_str = 'cm'
joule = Dimensionality({pq.kg: 1, pq.m: 2, pq.s: -2})
joule_str = 'kg*m**2/s**2'
joule_uni = 'kg·m²/s²'
joule_tex = r'$\mathrm{\frac{kg{\cdot}m^{2}}{s^{2}}}$'
joule_htm = 'kg&sdot;m<sup>2</sup>/s<sup>2</sup>'
Joule = Dimensionality({pq.J: 1})
Joule_str = 'J'

class TestDimensionality(TestCase):

    def test_dimensionality_str(self):
        self.assertEqual(str(meter), meter_str)
        self.assertEqual(joule.string, joule_str)
        self.assertEqual(joule.unicode, joule_uni)
        self.assertEqual(joule.latex, joule_tex)
        self.assertEqual(joule.html, joule_htm)
        self.assertEqual(Joule.string, 'J')

    def test_equality(self):
        self.assertTrue(meter == meter)
        self.assertTrue(joule == joule)
        self.assertFalse(meter == Joule)
        self.assertFalse(joule == Joule)

    def test_inequality(self):
        self.assertFalse(meter != meter)
        self.assertFalse(joule != joule)
        self.assertTrue(meter != Joule)
        self.assertTrue(joule != Joule)

    def test_copy(self):
        temp = meter.copy()
        self.assertTrue(temp is not meter)
        self.assertTrue(isinstance(temp, Dimensionality))
        self.assertTrue(temp == meter)
        temp[pq.m] += 1
        self.assertFalse(temp == meter)

    def test_addition(self):
        self.assertTrue(meter + meter is not meter)
        self.assertRaises(ValueError, op.add, meter, joule)
        self.assertRaises(ValueError, op.add, Joule, joule)
        self.assertRaises(TypeError, op.add, Joule, 0)
        self.assertRaises(TypeError, op.add, 0, joule)

    def test_inplace_addition(self):
        temp = meter.copy()
        temp += meter
        self.assertEqual(temp, meter)
        self.assertRaises(ValueError, op.iadd, meter, joule)
        self.assertRaises(ValueError, op.iadd, Joule, joule)
        self.assertRaises(TypeError, op.iadd, Joule, 0)
        self.assertRaises(TypeError, op.iadd, 0, joule)

    def test_subtraction(self):
        self.assertTrue(meter - meter is not meter)
        self.assertRaises(ValueError, op.sub, meter, joule)
        self.assertRaises(ValueError, op.sub, Joule, joule)
        self.assertRaises(TypeError, op.sub, Joule, 0)
        self.assertRaises(TypeError, op.sub, 0, joule)

    def test_inplace_subtraction(self):
        temp = meter.copy()
        temp -= meter
        self.assertEqual(temp, meter)
        self.assertRaises(ValueError, op.isub, meter, joule)
        self.assertRaises(ValueError, op.isub, Joule, joule)
        self.assertRaises(TypeError, op.isub, Joule, 0)
        self.assertRaises(TypeError, op.isub, 0, joule)

    def test_multiplication(self):
        self.assertEqual(meter*meter, Dimensionality({pq.m: 2}))
        self.assertEqual(meter*centimeter, Dimensionality({pq.m: 1, pq.cm: 1}))
        self.assertEqual(joule*meter, Dimensionality({pq.kg: 1, pq.m: 3, pq.s: -2}))
        self.assertRaises(TypeError, op.mul, Joule, 0)
        self.assertRaises(TypeError, op.mul, 0, joule)

    def test_inplace_multiplication(self):
        temp = meter.copy()
        temp *= meter
        self.assertEqual(temp, meter*meter)
        temp *= centimeter
        self.assertEqual(temp, meter*meter*centimeter)
        temp *= centimeter**-1
        self.assertEqual(temp, meter*meter)
        self.assertRaises(TypeError, op.imul, Joule, 0)
        self.assertRaises(TypeError, op.imul, 0, joule)

    def test_division(self):
        self.assertEqual(meter/meter, Dimensionality())
        self.assertEqual(joule/meter, Dimensionality({pq.kg: 1, pq.m: 1, pq.s: -2}))
        self.assertRaises(TypeError, op.truediv, Joule, 0)
        self.assertRaises(TypeError, op.truediv, 0, joule)

    def test_inplace_division(self):
        temp = meter.copy()
        temp /= meter
        self.assertEqual(temp, meter/meter)
        temp /= centimeter
        self.assertEqual(temp, meter/meter/centimeter)
        temp /= centimeter**-1
        self.assertEqual(temp, meter/meter)
        self.assertRaises(TypeError, op.itruediv, Joule, 0)
        self.assertRaises(TypeError, op.itruediv, 0, joule)

    def test_power(self):
        self.assertEqual(meter**2, meter*meter)
        self.assertEqual(meter**0, Dimensionality())
        self.assertEqual(joule**2, Dimensionality({pq.kg: 2, pq.m: 4, pq.s: -4}))
        self.assertRaises(TypeError, op.pow, Joule, joule)
        self.assertRaises(TypeError, op.pow, joule, Joule)
        self.assertEqual(meter**-1 == meter**-2, False)

    def test_inplace_power(self):
        temp = meter.copy()
        temp **= 2
        self.assertEqual(temp, meter**2)
        temp = joule.copy()
        temp **= 2
        self.assertEqual(temp, joule**2)
        temp = meter.copy()
        temp **= 0
        self.assertEqual(temp, Dimensionality())
        self.assertRaises(TypeError, op.ipow, Joule, joule)
        self.assertRaises(TypeError, op.ipow, joule, Joule)

    def test_simplification(self):
        self.assertEqual(Joule.simplified.string, 'kg*m**2/s**2')
        self.assertEqual(Joule.simplified, joule)


    def test_gt(self):
        self.assertTrue(joule > meter)
        self.assertTrue(Joule > meter)
        self.assertFalse(meter > joule)
        self.assertFalse(meter > Joule)
        self.assertFalse(joule > joule)
        self.assertFalse(joule > Joule)

    def test_ge(self):
        self.assertTrue(joule >= meter)
        self.assertTrue(Joule >= meter)
        self.assertFalse(meter >= joule)
        self.assertFalse(meter >= Joule)
        self.assertTrue(joule >= joule)
        self.assertTrue(joule >= Joule)

    def test_lt(self):
        self.assertTrue(meter < joule)
        self.assertTrue(meter < Joule)
        self.assertFalse(joule < meter)
        self.assertFalse(Joule < meter)
        self.assertFalse(joule < joule)
        self.assertFalse(Joule < joule)

    def test_le(self):
        self.assertTrue(meter <= joule)
        self.assertTrue(meter <= Joule)
        self.assertFalse(joule <= meter)
        self.assertFalse(Joule <= meter)
        self.assertTrue(joule <= joule)
        self.assertTrue(joule <= Joule)
