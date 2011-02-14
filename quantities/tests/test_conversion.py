# -*- coding: utf-8 -*-

from .. import units as pq
from .common import TestCase


class TestConversion(TestCase):

    def test_inplace_conversion(self):
        for u in ('ft', 'feet', pq.ft):
            q = 10*pq.m
            q.units = u
            self.assertQuantityEqual(q, 32.80839895 * pq.ft)

    def test_rescale(self):
        for u in ('ft', 'feet', pq.ft):
            self.assertQuantityEqual((10*pq.m).rescale(u), 32.80839895 * pq.ft)

    def test_compound_reduction(self):
        pc_per_cc = pq.CompoundUnit("pc/cm**3")
        temp = pc_per_cc * pq.CompoundUnit('m/m**3')

        self.assertQuantityEqual(
            temp.simplified,
            3.08568025e+22 / pq.m**4,
            delta=1e17
            )

        self.assertQuantityEqual(
            temp.rescale('pc**-4'),
            2.79740021556e+88 / pq.pc**4,
            delta=1e83
            )


class TestDefaultUnits(TestCase):

    def test_default_length(self):
        pq.set_default_units(length='mm')
        self.assertQuantityEqual(pq.m.simplified, 1000*pq.mm)

        pq.set_default_units(length='m')
        self.assertQuantityEqual(pq.m.simplified, pq.m)
        self.assertQuantityEqual(pq.mm.simplified, 0.001*pq.m)

    def test_default_system(self):
        pq.set_default_units('cgs')
        self.assertQuantityEqual(pq.kg.simplified, 1000*pq.g)
        self.assertQuantityEqual(pq.m.simplified, 100*pq.cm)

        pq.set_default_units('SI')
        self.assertQuantityEqual(pq.g.simplified, 0.001*pq.kg)
        self.assertQuantityEqual(pq.mm.simplified, 0.001*pq.m)

        pq.set_default_units('cgs', length='mm')
        self.assertQuantityEqual(pq.kg.simplified, 1000*pq.g)
        self.assertQuantityEqual(pq.m.simplified, 1000*pq.mm)
