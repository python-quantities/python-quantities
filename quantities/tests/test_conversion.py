# -*- coding: utf-8 -*-

import unittest
from .. import units as pq
from .. import quantity
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
            
    def test_rescale_preferred(self):
        quantity.PREFERRED = [pq.mV, pq.pA]
        q = 10*pq.V
        self.assertQuantityEqual(q.rescale_preferred(), q.rescale(pq.mV))
        q = 5*pq.A
        self.assertQuantityEqual(q.rescale_preferred(), q.rescale(pq.pA))
        quantity.PREFERRED = []
    
    def test_rescale_preferred_failure(self):
        quantity.PREFERRED = [pq.pA]
        q = 10*pq.V
        try:
            self.assertQuantityEqual(q.rescale_preferred(), q.rescale(pq.mV))
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        quantity.PREFERRED = []
    
    def test_rescale_noargs(self):
        quantity.PREFERRED = [pq.mV, pq.pA]
        q = 10*pq.V
        self.assertQuantityEqual(q.rescale(), q.rescale(pq.mV))
        q = 5*pq.A
        self.assertQuantityEqual(q.rescale(), q.rescale(pq.pA))
        quantity.PREFERRED = []
    
    def test_rescale_noargs_failure(self):
        quantity.PREFERRED = [pq.pA]
        q = 10*pq.V
        try:
            self.assertQuantityEqual(q.rescale_preferred(), q.rescale(pq.mV))
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        quantity.PREFERRED = []

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

class TestUnitInformation(TestCase):

    def test_si(self):
        pq.set_default_units(information='B')
        self.assertQuantityEqual(pq.kB.simplified, pq.B*pq.kilo)
        self.assertQuantityEqual(pq.MB.simplified, pq.B*pq.mega)
        self.assertQuantityEqual(pq.GB.simplified, pq.B*pq.giga)
        self.assertQuantityEqual(pq.TB.simplified, pq.B*pq.tera)
        self.assertQuantityEqual(pq.PB.simplified, pq.B*pq.peta)
        self.assertQuantityEqual(pq.EB.simplified, pq.B*pq.exa)
        self.assertQuantityEqual(pq.ZB.simplified, pq.B*pq.zetta)
        self.assertQuantityEqual(pq.YB.simplified, pq.B*pq.yotta)

    def test_si_aliases(self):
        prefixes = ['kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'yotta']
        for prefix in prefixes:
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'bytes'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octet'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octets'))

    def test_iec(self):
        pq.set_default_units(information='B')
        self.assertQuantityEqual(pq.KiB.simplified, pq.B*pq.kibi)
        self.assertQuantityEqual(pq.MiB.simplified, pq.B*pq.mebi)
        self.assertQuantityEqual(pq.GiB.simplified, pq.B*pq.gibi)
        self.assertQuantityEqual(pq.TiB.simplified, pq.B*pq.tebi)
        self.assertQuantityEqual(pq.PiB.simplified, pq.B*pq.pebi)
        self.assertQuantityEqual(pq.EiB.simplified, pq.B*pq.exbi)
        self.assertQuantityEqual(pq.ZiB.simplified, pq.B*pq.zebi)
        self.assertQuantityEqual(pq.YiB.simplified, pq.B*pq.yobi)

    def test_iec_aliases(self):
        prefixes = ['kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi', 'zebi', 'yobi']
        for prefix in prefixes:
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'bytes'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octet'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octets'))
