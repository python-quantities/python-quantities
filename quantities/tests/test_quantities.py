
import unittest
import tempfile
import shutil
import os
import numpy
import quantities as q
from quantities.quantity import ProtectedUnitsError

def test():
    assert 1==1, 'assert 1==1'

class TestQuantities(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(str(q.m), "1.0 m", str(q.m))
        self.assertEqual(str(q.J), "1.0 kg * m^2 / s^2", str(q.J))
        self.assertEqual(str(q.Quantity(1.0, q.J)), "1.0 kg * m^2 / s^2",
                         str(q.Quantity(1.0, q.J)))
        self.assertEqual(str(q.Quantity(1.0, 'J')), "1.0 kg * m^2 / s^2",
                         str(q.Quantity(1.0, 'J')))

    def test_units_protected(self):
        self.assertRaises(ProtectedUnitsError, q.m._set_units, 'ft')
        self.assertRaises(ProtectedUnitsError, q.m.modify_units, 'ft')
        self.assertRaises(ProtectedUnitsError, q.m.simplify_units)

    def test_unit_aggregation(self):
        self.assertEqual(str(q.J/q.m), "1.0 kg * m / s^2", str(q.J/q.m))
        self.assertEqual(str(q.J*q.m), "1.0 kg * m^3 / s^2", str(q.J*q.m))
        self.assertEqual(str(q.J*q.compound("parsec/cm^3")),
                         "1.0 kg * m^2 * (parsec/cm^3) / s^2",
                         str(q.J*q.compound("parsec/cm^3")))
        temp = 1.0*q.m
        temp.units = q.ft
        temp = q.compound("parsec/cm^3")*q.compound("m^3/m^2")
        temp.simplify_units()
        self.assertEqual(str(temp), "3.085678e+22 1 / m", str(temp))

    def test_ratios(self):
        self.assertAlmostEqual(q.m/q.ft, 3.280839895, 10, q.m/q.ft)
        self.assertAlmostEqual(q.J/q.BTU, 0.00094781712, 10, q.J/q.BTU)
        btu = q.compound('Btu')
        btu.simplify_units()
        self.assertAlmostEqual(q.J/btu, 0.00094781712, 10, q.J/btu)

    def test_compound_reduction(self):
        temp = q.compound('parsec/cm^3')*q.compound('m/m^3')
        self.assertEqual(str(temp), "1.0 (m/m^3) * (parsec/cm^3)", str(temp))
        temp.simplify_units()
        temp.units=q.parsec**-4
        self.assertEqual(str(temp), "2.7973900166e+88 1 / parsec^4", str(temp))
        temp.units=q.m**-4
        self.assertEqual(str(temp), "3.085678e+22 1 / m^4", str(temp))
        self.assertEqual(str(1/temp), "3.2407788499e-23 m^4", str(1/temp))
        self.assertEqual(str(temp**-1), "3.2407788499e-23 m^4", str(temp**-1))

