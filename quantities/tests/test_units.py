import pytest

from .. import units as pq
from ..quantity import Quantity
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

    def test_code_injection(self):
        with pytest.raises(LookupError) as exc_info:
            pq.CompoundUnit("exec(\"print('Hello there.')\\nprint('General Wasabi!')\")")
            assert "Wasabi" in str(exc_info.value)


class TestForceUnits(TestCase):

    def test_pond_newtons(self):
        # A pond is a gram-force, not a kilogram-force.
        self.assertQuantityEqual(
            pq.pond.rescale(pq.N), 0.00980665 * pq.N, delta=1e-12
        )

    def test_pond_names(self):
        self.assertEqual(pq.pond.name, 'pond')
        self.assertEqual(pq.pond.symbol, 'p')
        for unit in ('pond', 'ponds', 'p'):
            with self.subTest(unit=unit):
                self.assertQuantityEqual(
                    Quantity(1, unit).rescale(pq.N),
                    0.00980665 * pq.N,
                    delta=1e-12
                )

    def test_pond_array(self):
        self.assertQuantityEqual(
            ([-1, 0, 1, 1000] * pq.pond).rescale(pq.N),
            [-0.00980665, 0, 0.00980665, 9.80665] * pq.N,
            delta=1e-12
        )

    def test_pond_dimensionality(self):
        self.assertEqual(
            pq.pond.simplified.dimensionality,
            (pq.kg * pq.m / pq.s**2).dimensionality
        )

    def test_gram_force_newtons(self):
        self.assertQuantityEqual(
            pq.gram_force.rescale(pq.N), 0.00980665 * pq.N, delta=1e-12
        )

    def test_kilogram_force_newtons(self):
        self.assertQuantityEqual(
            pq.kgf.rescale(pq.N), 9.80665 * pq.N, delta=1e-12
        )
