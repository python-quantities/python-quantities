# -*- coding: utf-8 -*-

from .. import units as pq
from .common import TestCase

class TestUnits(TestCase):

    def test_compound_units(self):
        pc_per_cc = pq.CompoundUnit("pc/cm**3")
        self.assertEqual(str(pc_per_cc.dimensionality), "(pc/cm**3)")
        self.assertEqual(str(pc_per_cc), "1 (pc/cm**3)")

        temp = pc_per_cc * pq.CompoundUnit('m/m**3')
        self.assertEqual(str(temp.dimensionality), "(pc/cm**3)*(m/m**3)")
        self.assertEqual(str(temp), "1.0 (pc/cm**3)*(m/m**3)")

    def test_units_protected(self):
        def setunits(u, v):
            u.units = v

        def inplace(op, u, val):
            getattr(u, '__i%s__'%op)(val)

        self.assertRaises(AttributeError, setunits, pq.m, pq.ft)
        self.assertRaises(TypeError, inplace, 'add', pq.m, pq.m)
        self.assertRaises(TypeError, inplace, 'sub', pq.m, pq.m)
        self.assertRaises(TypeError, inplace, 'mul', pq.m, pq.m)
        self.assertRaises(TypeError, inplace, 'truediv', pq.m, pq.m)
        self.assertRaises(TypeError, inplace, 'pow', pq.m, 2)

    def test_units_copy(self):
        self.assertQuantityEqual(pq.m.copy(), pq.m)
        pc_per_cc = pq.CompoundUnit("pc/cm**3")
        self.assertQuantityEqual(pc_per_cc.copy(), pc_per_cc)
