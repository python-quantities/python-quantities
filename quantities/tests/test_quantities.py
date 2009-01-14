# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
import numpy
from quantities import *
import quantities as q
from nose.tools import *

def test_immutabledimensionality_iter():
    assert_equal(str([i for i in m.dimensionality]), '[1 m (meter)]')
    assert_equal(str([i for i in m.dimensionality.iterkeys()]), '[1 m (meter)]')

def test_immutabledimensionality_copy():
    assert_equal(m.dimensionality, m.dimensionality.copy())

def test_immutabledimensionality_get():
    assert_equal(m.dimensionality.get(m), 1)
    assert_equal(m.dimensionality.get(ft, 2), 2)
    assert_true(m in m.dimensionality)

def test_units_protected():
    def setunits(u, v):
        u.units = v
    def inplace(op, u, val):
        getattr(u, '__i%s__'%op)(val)
    assert_raises(AttributeError, setunits, m, ft)
    assert_raises(TypeError, inplace, 'add', m, m)
    assert_raises(TypeError, inplace, 'sub', m, m)
    assert_raises(TypeError, inplace, 'mul', m, m)
    assert_raises(TypeError, inplace, 'div', m, m)
    assert_raises(TypeError, inplace, 'truediv', m, m)
    assert_raises(TypeError, inplace, 'pow', m, 2)

def test_quantity_creation():
    assert_raises(LookupError, Quantity, 1, 'nonsense')
    assert_equal(str(Quantity(1, '')), '1.0 dimensionless')

def test_scalar_equality():
    assert_true(J == J)
    assert_true(1*J == J)
    assert_true(str(1*J) == str(J))
    assert_true(J == q.kg*q.m**2/q.s**2)

    assert_false(J == erg)
    assert_false(2*J == J)
    assert_false(str(2*J) == str(J))
    assert_false(J == 2*q.kg*q.m**2/q.s**2)

    def eq(q1, q2):
        return q1 == q2
    assert_raises(ValueError, eq, J, kg)

def test_scalar_inequality():
    assert_true(J != erg)
    assert_true(2*J != J)
    assert_true(str(2*J) != str(J))
    assert_true(J != 2*q.kg*q.m**2/q.s**2)

    assert_false(J != J)
    assert_false(1*J != J)
    assert_false(str(1*J) != str(J))
    assert_false(J != 1*q.kg*q.m**2/q.s**2)

def test_scalar_comparison():
    assert_true(2*J > J)
    assert_true(2*J > 1*J)
    assert_true(1*J >= J)
    assert_true(1*J >= 1*J)
    assert_true(2*J >= J)
    assert_true(2*J >= 1*J)

    assert_true(0.5*J < J)
    assert_true(0.5*J < 1*J)
    assert_true(0.5*J <= J)
    assert_true(0.5*J <= 1*J)
    assert_true(1.0*J <= J)
    assert_true(1.0*J <= 1*J)

    assert_false(2*J < J)
    assert_false(2*J < 1*J)
    assert_false(2*J <= J)
    assert_false(2*J <= 1*J)

    assert_false(0.5*J > J)
    assert_false(0.5*J > 1*J)
    assert_false(0.5*J >= J)
    assert_false(0.5*J >= 1*J)

def test_array_equality():
    assert_false(
        str(Quantity([1, 2, 3, 4], 'J')) == str(Quantity([1, 22, 3, 44], 'J'))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')) == str(Quantity([1, 2, 3, 4], 'J'))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==Quantity([1, 22, 3, 44], 'J')) == \
            str(numpy.array([True, False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==Quantity([1, 22, 3, 44], J)) == \
            str(numpy.array([True, False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==[1, 22, 3, 44]*J) == \
            str(numpy.array([True, False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==numpy.array([1, 22, 3, 44])*J) == \
            str(numpy.array([True, False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==\
            Quantity(numpy.array([1, 22, 3, 44]), 'J')) == \
            str(numpy.array([True, False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==\
            Quantity(Quantity([1, 22, 3, 44], 'J'))) == \
            str(numpy.array([True, False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==[1, 22, 3, 44]*kg*m**2/s**2) == \
            str(numpy.array([True, False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')==Quantity([1, 22, 3, 44], 'J')) == \
            str(numpy.array([True, False, True, False]))
    )

def test_array_inequality():
    assert_true(
        str(Quantity([1, 2, 3, 4], 'J')!=Quantity([1, 22, 3, 44], 'J')) == \
            str(numpy.array([False, True, False, True]))
    )

def test_array_comparison():
    assert_true(
        str(Quantity([1, 2, 33], 'J')>Quantity([1, 22, 3], 'J')) == \
            str(numpy.array([False, False, True]))
    )
    assert_true(
        str(Quantity([1, 2, 33], 'J')>=Quantity([1, 22, 3], 'J')) == \
            str(numpy.array([True, False, True]))
    )
    assert_true(
        str(Quantity([1, 2, 33], 'J')<Quantity([1, 22, 3], 'J')) == \
            str(numpy.array([False, True, False]))
    )
    assert_true(
        str(Quantity([1, 2, 33], 'J')<=Quantity([1, 22, 3], 'J')) == \
            str(numpy.array([True, True, False]))
    )

def test_uncertainquantity_creation():
    a = UncertainQuantity(1, m)
    assert_equal(str(a), '1.0 m\n+/-0.0 m (1 sigma)')
    a = UncertainQuantity([1, 1, 1], m)
    assert_equal(str(a), '[ 1.  1.  1.] m\n+/-[ 0.  0.  0.] m (1 sigma)')
    a = UncertainQuantity(a)
    assert_equal(str(a), '[ 1.  1.  1.] m\n+/-[ 0.  0.  0.] m (1 sigma)')
    a = UncertainQuantity([1, 1, 1], m, [.1, .1, .1])
    assert_equal(str(a), '[ 1.  1.  1.] m\n+/-[ 0.1  0.1  0.1] m (1 sigma)')
    assert_raises(ValueError, UncertainQuantity, [1, 1, 1], m, 1)
    assert_raises(ValueError, UncertainQuantity, [1, 1, 1], m, [1, 1])

def test_uncertainquantity_set_units():
    a = UncertainQuantity([1, 1, 1], m, [.1, .1, .1])
    a.units = ft
    assert_equal(
        str(a),
        '[ 3.2808399  3.2808399  3.2808399] ft'
        '\n+/-[ 0.32808399  0.32808399  0.32808399] ft (1 sigma)'
    )

def test_uncertainquantity_rescale():
    a = UncertainQuantity([1, 1, 1], m, [.1, .1, .1])
    b = a.rescale(ft)
    assert_equal(
        str(b),
        '[ 3.2808399  3.2808399  3.2808399] ft'
        '\n+/-[ 0.32808399  0.32808399  0.32808399] ft (1 sigma)'
    )

def test_uncertainquantity_simplified():
    a = 1000*eV
    assert_equal(
        str(a.simplified),
        '1.602176487e-16 kg·m²/s²\n+/-4e-24 kg·m²/s² (1 sigma)'
    )

def test_uncertainquantity_set_uncertainty():
    a = UncertainQuantity([1, 2], 'm', [.1, .2])
    assert_equal(
        str(a),
        '[ 1.  2.] m\n+/-[ 0.1  0.2] m (1 sigma)'
    )
    a.uncertainty = [1., 2.]
    assert_equal(
        str(a),
        '[ 1.  2.] m\n+/-[ 1.  2.] m (1 sigma)'
    )
    def set_u(q, u):
        q.uncertainty = u
    assert_raises(ValueError, set_u, a, 1)


def test_uncertainquantity_multiply():
    a = UncertainQuantity([1, 2], 'm', [.1, .2])
    assert_equal(
        str(a*a),
        '[ 1.  4.] m²\n+/-[ 0.14142136  0.56568542] m² (1 sigma)'
    )
    assert_equal(
        str(a*2),
        '[ 2.  4.] m\n+/-[ 0.2  0.4] m (1 sigma)'
    )

def test_uncertainquantity_divide():
    a = UncertainQuantity([1, 2], 'm', [.1, .2])
    assert_equal(
        str(a/a),
        '[ 1.  1.] dimensionless\n+/-[ 0.14142136  0.14142136] '
        'dimensionless (1 sigma)'
    )
    assert_equal(
        str(a/2),
        '[ 0.5  1. ] m\n+/-[ 0.05  0.1 ] m (1 sigma)'
    )

class TestQuantities(unittest.TestCase):

    def numAssertEqual(self, a1, a2):
        """Test for equality of numarray fields a1 and a2.
        """
        self.assertEqual(a1.shape, a2.shape)
        self.assertEqual(a1.dtype, a2.dtype)
        self.assertTrue((a1 == a2).all())

    def numAssertAlmostEqual(self, a1, a2):
        """Test for approximately equality of numarray fields a1 and a2.
        """
        self.assertEqual(a1.shape, a2.shape)
        self.assertEqual(a1.dtype, a2.dtype)
        if a1.dtype == 'Float64' or a1.dtype == 'Complex64':
            prec = 15
        else:
            prec = 7
        # the complex part of this does not function correctly and will throw
        # errors that need to be fixed if it is to be used
        if numpy.iscomplex(a1).all():
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

    def test_simple(self):
        self.assertEqual(str(q.m), "1.0 m", str(q.m))
        self.assertEqual(str(q.J), "1.0 J", str(q.J))

    def test_creation(self):
        self.numAssertEqual(
            [100, -1.02, 30] * q.cm**2,
            q.Quantity(numpy.array([100, -1.02, 30]),
            q.cm**2)
        )
        self.assertEqual(
            str([100, -1.02, 30] * q.cm**2),
            str(q.Quantity(numpy.array([100, -1.02, 30]), q.cm**2))
        )

        self.assertEqual(
            -10.1 * q.ohm,
            q.Quantity(-10.1, q.ohm)
        )

        self.assertEqual(
            str(-10.1 * q.ohm),
            str(q.Quantity(-10.1, q.ohm))
        )

    def test_unit_aggregation(self):
        joule = q.kg*q.m**2/q.s**2
        pc_per_cc = q.CompoundUnit("pc/cm**3")
        area_per_volume = q.CompoundUnit("m**2/m**3")
        self.assertEqual(str(joule/q.m), "1.0 kg·m/s²", str(joule/q.m))
        self.assertEqual(str(joule*q.m), "1.0 kg·m³/s²", str(joule*q.m))
        self.assertEqual(
            str(q.J*pc_per_cc),
            "1.0 J·(pc/cm³)",
            str(q.J*pc_per_cc)
        )
        temp = pc_per_cc / area_per_volume
        self.assertEqual(
            str(temp.simplified),
            "3.08568025e+22 1/m",
            str(temp.simplified)
        )

    def test_ratios(self):
        self.assertAlmostEqual(
            q.m/q.ft.rescale(q.m),
            3.280839895,
            10,
            q.m/q.ft.rescale(q.m)
        )
        self.assertAlmostEqual(
            q.J/q.BTU.rescale(q.J),
            0.00094781712,
            10,
            q.J/q.BTU.rescale(q.J)
        )

    def test_compound_reduction(self):
        pc_per_cc = q.CompoundUnit("pc/cm**3")
        temp = pc_per_cc * q.CompoundUnit('m/m**3')
        self.assertEqual(str(temp), "1.0 (pc/cm³)·(m/m³)", str(temp))
        temp = temp.simplified
        temp.units=q.pc**-4
        self.assertEqual(str(temp), "2.79740021556e+88 1/pc⁴", str(temp))
        temp.units=q.m**-4
        self.assertEqual(str(temp), "3.08568025e+22 1/m⁴", str(temp))
        self.assertEqual(str(1/temp), "3.24077648681e-23 m⁴", str(1/temp))
        self.assertEqual(
            str(temp**-1),
            "3.24077648681e-23 m⁴",
            str(temp**-1)
        )

        # does this handle regular units correctly?
        temp1 = 3.14159 * q.m

        self.assertAlmostEqual(temp1, temp1.simplified)

        self.assertEqual(str(temp1), str(temp1.simplified))

    def test_equality(self):
        test1 = 1.5 * q.km
        test2 = 1.5 * q.km

        self.assertEqual(test1, test2)
        test2.units = q.ft

        self.assertAlmostEqual(test1, test2.rescale(q.km))

        # test less than and greater than
        self.assertTrue(1.5 * q.km > 2.5 * q.cm)
        self.assertTrue(1.5 * q.km >= 2.5 * q.cm)
        self.assertTrue(not (1.5 * q.km < 2.5 * q.cm))
        self.assertTrue(not (1.5 * q.km <= 2.5 * q.cm))

        self.assertTrue(
            1.5 * q.km != 1.5 * q.cm,
            "unequal quantities are not-not-equal"
        )

    def test_addition(self):
        # arbitrary test of addition
        self.assertAlmostEqual((5.2 * q.eV) + (300.2 * q.eV), 305.4 * q.eV, 5)

        # test of addition using different units
        self.assertAlmostEqual(
            (5 * q.hp + 7.456999e2 * q.W.rescale(q.hp)),
            (6 * q.hp)
        )

        def add_bad_units():
            """just a function that raises an incompatible units error"""
            return (1 * q.kPa) + (5 * q.lb)

        self.assertRaises(ValueError, add_bad_units)

        # add a scalar and an array
        arr = numpy.array([1,2,3,4,5])
        temp1 = arr * q.rem
        temp2 = 5.5 * q.rem

        self.assertEqual(
            str(temp1 + temp2),
            "[  6.5   7.5   8.5   9.5  10.5] rem"
        )
        self.assertTrue(((arr+5.5) * q.rem == temp1 + temp2).all())

        # with different units
        temp4 = 1e-2 * q.sievert
        self.numAssertAlmostEqual(
            temp1 + temp4.rescale(q.rem),
            temp1 + 1 * q.rem
        )

        # add two arrays
        temp3 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * q.rem

        self.assertEqual(
            str(temp1 + temp3),
            "[  6.5   8.5   8.5   9.5  10.5] rem"
        )
        # two arrays with different units
        temp5 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * q.sievert

        self.assertEqual(
            str(temp1 + temp5.rescale(q.rem)),
            "[  6.5   8.5   8.5   9.5  10.5] rem"
        )

        # in-place addition
        temp1 = 1*m
        temp2 = 1*m
        temp1+=temp1
        self.assertEqual(str(temp1), str(temp2+temp2))

        temp1 = [1, 2, 3, 4]*m
        temp2 = [1, 2, 3, 4]*m
        temp1+=temp1
        self.assertEqual(str(temp1), str(temp2+temp2))

        def iadd(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, iadd, 1*m, 1)

    def test_substraction(self):
        # arbitrary test of subtraction
        self.assertAlmostEqual((5.2 * q.eV) - (300.2 * q.eV), -295.0 * q.eV)

        # the formatting should be the same
        self.assertEqual(
            str((5.2 * q.energy.eV) - (300.2 * q.energy.eV)),
            str(-295.0 * q.energy.eV)
        )

        # test of subtraction using different units
        self.assertAlmostEqual(
            (5 * q.hp - 7.456999e2 * q.W.rescale(q.hp)),
            (4 * q.hp)
        )

        def subtract_bad_units():
            """just a function that raises an incompatible units error"""
            return (1 * q.kPa) - (5 * q.lb)

        self.assertRaises(ValueError, subtract_bad_units)

        # subtract a scalar and an array
        arr = numpy.array([1,2,3,4,5])
        temp1 = arr * q.rem
        temp2 = 5.5 * q.rem

        self.assertEqual(str(temp1 - temp2), "[-4.5 -3.5 -2.5 -1.5 -0.5] rem")
        self.numAssertEqual((arr-5.5) * q.rem, temp1 - temp2)

        # with different units
        temp4 = 1e-2 * q.sievert
        self.numAssertAlmostEqual(temp1 - temp4.rescale(q.rem), temp1 - q.rem)

        #subtract two arrays
        temp3 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * q.rem

        self.assertEqual(str(temp1 - temp3), "[-4.5 -4.5 -2.5 -1.5 -0.5] rem")
        #two arrays with different units
        temp5 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * q.sievert

        self.assertEqual(
            str(temp1 - temp5.rescale(q.rem)),
            "[-4.5 -4.5 -2.5 -1.5 -0.5] rem"
        )

        # in-place
        temp1 = 1*m
        temp2 = 1*m
        temp1-=temp1
        self.assertEqual(str(temp1), str(temp2-temp2))

        temp1 = [1, 2, 3, 4]*m
        temp2 = [1, 2, 3, 4]*m
        temp1-=temp1
        self.assertEqual(str(temp1), str(temp2-temp2))

        def isub(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, isub, temp1, 1)

    def test_multiplication(self):
        #arbitrary test of multiplication
        self.assertAlmostEqual(
            (10.3 * q.kPa) * (10 * q.inch),
            103.0 * q.kPa*q.inch
        )

        self.assertAlmostEqual((5.2 * q.eV) * (300.2 * q.eV), 1561.04 * q.eV**2)

        # the formatting should be the same
        self.assertEqual(
            str((10.3 * q.kPa) * (10 * q.inch)),
            str( 103.0 * q.kPa*q.inch)
        )
        self.assertEqual(
            str((5.2 * q.energy.eV) * (300.2 * q.energy.eV)),
            str(1561.04 * q.energy.eV**2)
        )

        # does multiplication work with arrays?
        # multiply an array with a scalar
        temp1  = numpy.array ([3,4,5,6,7]) * q.J
        temp2 = .5 * q.s**-1

        self.assertEqual(
            str(temp1 * temp2),
            "[ 1.5  2.   2.5  3.   3.5] J/s"
        )

        # multiply an array with an array
        temp3 = numpy.array ([4,4,5,6,7]) * q.s**-1
        self.assertEqual(
            str(temp1 * temp3),
            "[ 12.  16.  25.  36.  49.] J/s"
        )

        # in-place
        temp1 = 1*m
        temp2 = 1*m
        temp1 *= temp1
        self.assertEqual(str(temp1), str(temp2*temp2))

        temp1 = [1, 2, 3, 4]*m
        temp2 = [1, 2, 3, 4]*m
        temp1 *= temp1
        self.assertEqual(str(temp1), str(temp2*temp2))

    def test_division(self):
        #arbitrary test of division
        self.assertAlmostEqual(
            (10.3 * q.kPa) / (1 * q.inch),
            10.3 * q.kPa/q.inch
        )

        self.assertAlmostEqual(
            (5.2 * q.eV) / (400.0 * q.eV),
            q.Quantity(.013)
        )

        # the formatting should be the same
        self.assertEqual(
            str((5.2 * q.energy.eV) / (400.0 * q.energy.eV)),
            str(q.Quantity(.013))
        )

        # divide an array with a scalar
        temp1  = numpy.array ([3,4,5,6,7]) * q.J
        temp2 = .5 * q.s**-1

        self.assertEqual(
            str(temp1 / temp2),
            "[  6.   8.  10.  12.  14.] s·J"
        )

        # divide an array with an array
        temp3 = numpy.array([4,4,5,6,7]) * q.s**-1
        self.assertEqual(
            str(temp1 / temp3),
            "[ 0.75  1.    1.    1.    1.  ] s·J"
        )

        # in-place
        temp1 = 1*m
        temp2 = 1*m
        temp1 /= temp1
        self.assertEqual(str(temp1), str(temp2/temp2))

        temp1 = [1, 2, 3, 4]*m
        temp2 = [1, 2, 3, 4]*m
        temp1 /= temp1
        self.assertEqual(str(temp1), str(temp2/temp2))

    def test_powering(self):
        # test raising a quantity to a power
        self.assertAlmostEqual((5.5 * q.cm)**5, (5.5**5) * (q.cm**5))
        self.assertEqual(str((5.5 * q.cm)**5), str((5.5**5) * (q.cm**5)))

        # must also work with compound units
        self.assertAlmostEqual((5.5 * q.J)**5, (5.5**5) * (q.J**5))
        self.assertEqual(str((5.5 * q.J)**5), str((5.5**5) * (q.J**5)))

        # does powering work with arrays?
        temp = numpy.array([1, 2, 3, 4, 5]) * q.kg
        temp2 = (numpy.array([1, 8, 27, 64, 125]) **2) * q.kg**6

        self.assertEqual(
            str(temp**3),
            "[   1.    8.   27.   64.  125.] kg³"
        )
        self.assertEqual(str(temp**6), str(temp2))

        def q_pow_r(q1, q2):
            return q1 ** q2

        self.assertRaises(ValueError, q_pow_r, 10.0 * q.m, 10 * q.J)
        self.assertRaises(ValueError, q_pow_r, 10.0 * q.m, numpy.array([1, 2, 3]))

        self.assertEqual( (10 * q.J) ** (2 * q.J/q.J) , 100 * q.J**2 )

        # test rpow here
        self.assertRaises(ValueError, q_pow_r, 10.0, 10 * q.J)

        self.assertEqual(10**(2*q.J/q.J), 100)

        # in-place
        temp1 = 1*m
        temp2 = 1*m
        temp1 **= 2
        self.assertEqual(str(temp1), str(temp2*temp2))

        temp1 = [1, 2, 3, 4]*m
        temp2 = [1, 2, 3, 4]*m
        temp1 **= 2
        self.assertEqual(str(temp1), str(temp2*temp2))

        def ipow(q1, q2):
            q1 -= q2
        self.assertRaises(ValueError, ipow, 1*m, [1, 2])

    def test_getitem(self):
        tempArray1 = q.Quantity(numpy.array([1.5, 2.5 , 3, 5]), q.J)
        temp = 2.5 * q.J
        # check to see if quantities brought back from an array are good
        self.assertEqual(tempArray1[1], temp )
        # check the formatting
        self.assertEqual(str(tempArray1[1]), str(temp))

        def tempfunc(index):
            return tempArray1[index]

        # make sure indexing is correct
        self.assertRaises(IndexError, tempfunc, 10)

        # test get item using slicing
        tempArray2 = [100, .2, -1, -5, -6] * q.mA
        tempArray3 = [100, .2, -1, -5] * q.mA
        tempArray4 = [.2, -1 ] * q.mA

        self.numAssertEqual(tempArray2[:], tempArray2)

        self.numAssertEqual(tempArray2[:-1], tempArray3)
        self.numAssertEqual(tempArray2[1:3], tempArray4)

    def test_setitem (self):
        temp = q.Quantity([0,2,5,7.6], q.lb)

        # needs to check for incompatible units
        def test(value):
            temp[2] = value

        # make sure normal assignment works correctly
        test(2 *q.lb)

        self.assertRaises(ValueError, test, 60 * q.inch * q.J)
        # even in the case when the quantity has no units
        # (maybe this could go away)
        self.assertRaises(ValueError, test, 60)

        #test set item using slicing
        tempArray2 = [100, .2, -1, -5, -6] * q.mA
        tempArray3 = [100, .2, 0, 0, -6] * q.mA
        tempArray4 = [100,  1,  1,  1,  1] * q.mA

        tempArray4[1:] = [.2, -1, -5, -6] * q.mA
        self.numAssertEqual(tempArray4, tempArray2)

        tempArray3[2:4] = [-1, -5] * q.mA
        self.numAssertEqual(tempArray3, tempArray2)

        tempArray4[:] = [100, .2, -1, -5, -6] * q.mA
        self.numAssertEqual(tempArray4, tempArray2)

        # check and see that dimensionless numbers work correctly
        tempArray5 = q.Quantity([.2, -3, -5, -9,10])
        tempArray6 = q.Quantity([.2, -3, 0, 0,11])

        tempArray5[4] = 1 + tempArray5[4]
        tempArray5[2:4] = numpy.zeros(2)

        self.numAssertEqual(tempArray5, tempArray6)

        # needs to check for out of bounds
        def tempfunc(value):
            temp[10] = value

        self.assertRaises(IndexError, tempfunc, 5 * q.lb)

    def test_iterator(self):
        f = numpy.array([100, 200, 1, 60, -80])
        x = f * q.kPa

        # make sure the iterator objects have the correct units
        for i in x:
            # currently fails
            self.assertEqual(i.units, q.kPa.units)

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
        self.numAssertEqual(m.argsort(), numpy.array([0, 4, 1, 2, 3, 5]))

        # diagonal
        t = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * q.kPa
        self.numAssertEqual(t.diagonal(offset=1), [2, 3] * q.kPa)

        # compress
        self.numAssertEqual(z.compress(z > 5 * q.inch), [6, 6, 6] * q.inch)

        # searchsorted
        m.sort()
        self.numAssertEqual(m.searchsorted([5.5, 9.5] * q.MeV),
                            numpy.array([4,6]))

        def searchsortedError():
            m.searchsorted([1])

        # make sure the proper error is raised when called with improper units
        self.assertRaises(ValueError, searchsortedError)

        # nonzero
        j = [1, 0, 5, 6, 0, 9] * q.MeV
        self.numAssertEqual(j.nonzero()[0], numpy.array([0, 2, 3, 5]))

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
            numpy.sqrt(((1 - 201.520002)**2 + (3.00001 -201.520002)**2 + \
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

    def test_specific_bugs(self):
        # bug 1) where trying to modify units to incompatible ones
        temp = q.Quantity(1, q.lb)

        # needs to check for incompatible units
        def test(units):
            temp.units = units

        # this currently screws up q.J
        self.assertRaises(ValueError, test, q.inch * q.J)

        # check for this bug
        self.assertEqual(
            str(q.J.rescale('kg*m**2/s**2')),
            "1.0 kg·m²/s²"
        )
