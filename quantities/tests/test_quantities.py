
import unittest
import tempfile
import shutil
import os
import numpy
import quantities as q
from quantities.quantity import ProtectedUnitsError
from quantities.dimensionality import IncompatibleUnits

def test():
    assert 1==1, 'assert 1==1'

class TestQuantities(unittest.TestCase):

    def numAssertEqual(self, a1, a2):
        """Test for equality of numarray fields a1 and a2.
        """
        self.assertEqual(a1.shape, a2.shape)
        self.assertEqual(a1.dtype, a2.dtype)
        self.assertTrue(numpy.alltrue(numpy.equal(a1.flat, a2.flat)))

    def numAssertAlmostEqual(self, a1, a2):
        """Test for approximately equality of numarray fields a1 and a2.
        """
        self.assertEqual(a1.shape, a2.shape)
        self.assertEqual(a1.dtype, a2.dtype)
        if a1.dtype == 'Float64' or a1.dtype == 'Complex64':
            prec = 15
        else:
            prec = 7
        #the complex part of this does not function correctly and will throw errors that need to be fixed if it is to be used
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
        
        
        #does this handle regular units correctly?
        temp1 = 3.14159 * q.m
        
        temp2 = 3.14159 * q.m
        temp2.simplify_units() 
        #simplifying the units of a quantity with base units should not affect it
        # this currently fails
        self.assertAlmostEqual( temp1, temp2)

        self.assertEqual( str(temp1) , str(temp2))
        
    def test_equality(self):
        test1 = 1.5 * q.kilometer 
        test2 = 1.5 * q.kilometer
        
        self.assertEqual(test1, test2)
        test2.units = q.ft 
        
        self.assertAlmostEqual( test1, test2)
        # test less than and greater than
        
        self.assertTrue(1.5 * q.kilometer > 2.5 * q.cm)
        self.assertTrue(1.5 * q.kilometer >= 2.5 * q.cm)
        self.assertTrue( not (1.5 * q.kilometer < 2.5 * q.cm))
        self.assertTrue( not (1.5 * q.kilometer <= 2.5 * q.cm))
        
        self.assertTrue( 1.5 * q.kilometer != 1.5 * q.cm, "unequal quantities are not-not-equal")
         
        
    def test_addition(self):
        #arbitrary test of addition
        self.assertAlmostEqual ( (5.2 * q.eV) + (300.2 * q.eV) ,   305.4 * q.eV ,5 )
    
        # test addition with compound units 
        self.assertAlmostEqual ( (5.2 * q.eV_) + (300.2 * q.eV_) ,   305.4 * q.eV_ ,5)
        # the formatting should be the same 
        
        #test of addition using different units
        self.assertAlmostEqual( (5 * q.hp_ + 7.456999e2 *q.W),   (6 * q.hp_) )
        
        
        def add_bad_units():
            """just a function that raises an incompatible units error"""
            return ( 1 * q.kPa) + (5 * q.lb)
        
        self.assertRaises(IncompatibleUnits, add_bad_units)
  

        # does add work correctly with arrays?
        
        # add a scalar and an array
        arr = numpy.array([1,2,3,4,5])
        temp1 = arr * q.rem
        
        temp2 = 5.5 * q.rems
        
        self.assertEqual(str(temp1 + temp2),   "[  6.5   7.5   8.5   9.5  10.5] (rem)")
        self.assertTrue( ((arr+5.5) * q.rem == temp1 + temp2).all())
        # with different units
        
        temp4 = 1e-2 * q.sievert
        self.numAssertAlmostEqual( (temp1 + temp4), temp1 + 1 * q.rem)
        
        
        #add two arrays
        temp3 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * q.rem 
        
        self.assertEqual(str(temp1 + temp3),   "[  6.5   8.5   8.5   9.5  10.5] (rem)")
        #two arrays with different units
        temp5 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * q.sievert 
        
        self.assertEqual(str(temp1 + temp5),   "[  6.5   8.5   8.5   9.5  10.5] (rem)")
                 
        
    def test_substraction(self):
        #arbitrary test of subtraction
        self.assertAlmostEqual ( (5.2 * q.eV) - (300.2 * q.eV) ,   -295.0 * q.eV)
    
        # test subtraction with compound units 
        self.assertAlmostEqual ( (5.2 * q.eV_) - (300.2 * q.eV_) ,   -295.0 * q.eV_)
        # the formatting should be the same 
        self.assertEqual ( str(   (5.2 * q.eV_) - (300.2 * q.eV_) ), str(  -295.0 * q.eV_)   )
        
        #test of subtraction using different units
        self.assertAlmostEqual( (5 * q.hp_ - 7.456999e2 *q.W),   (4 * q.hp_) )
        
        
        def subtract_bad_units():
            """just a function that raises an incompatible units error"""
            return ( 1 * q.kPa) + (5 * q.lb)
        
        self.assertRaises(IncompatibleUnits, subtract_bad_units)
        
        

        # does subtraction work correctly with arrays?
        
        # subtract a scalar and an array
        arr = numpy.array([1,2,3,4,5])
        temp1 = arr * q.rem
        
        temp2 = 5.5 * q.rems
        
        self.assertEqual(str(temp1 - temp2),   "[-4.5 -3.5 -2.5 -1.5 -0.5] (rem)")
        self.numAssertEqual( (arr-5.5) * q.rem, temp1 - temp2)
        # with different units
        
        temp4 = 1e-2 * q.sievert
        self.numAssertAlmostEqual( (temp1 - temp4), temp1 - 1 * q.rem)
        
        
        #subtract two arrays
        temp3 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * q.rem 
        
        self.assertEqual(str(temp1 - temp3),   "[-4.5 -4.5 -2.5 -1.5 -0.5] (rem)")
        #two arrays with different units
        temp5 = numpy.array([5.5, 6.5, 5.5, 5.5, 5.5]) * 1e-2 * q.sievert 
        
        self.assertEqual(str(temp1 - temp5),   "[-4.5 -4.5 -2.5 -1.5 -0.5] (rem)")

        
    
    def test_multiplication(self):
        #arbitrary test of multiplication
        self.assertAlmostEqual(  (10.3 * q.kPa) * (10 * q.inch) , 103.0 * q.kPa*q.inch)
        
        self.assertAlmostEqual ( (5.2 * q.eV) * (300.2 * q.eV) ,   1561.04 * q.eV**2)
    
        # test multiplication with compound units 
        self.assertAlmostEqual ( (5.2 * q.eV_) * (300.2 * q.eV_) ,   1561.04 * q.eV_**2)
        # the formatting should be the same 
        self.assertEqual(   str( (10.3 * q.kPa) * (10 * q.inch) ), str( 103.0 * q.kPa*q.inch)   )
        self.assertEqual ( str(    (5.2 * q.eV_) * (300.2 * q.eV_)    ) , str(  1561.04 * q.eV_**2   )   )
        
        
        #does multiplication work with arrays?
        #multiply an array with a scalar
        
        temp1  = numpy.array ([3,4,5,6,7]) * q.J_
        temp2 = .5 * q.s**-1
        
        self.assertEqual( str ( temp1 * temp2) , "[ 1.5  2.   2.5  3.   3.5] (J) / s")

        # multiply an array with an array     
        temp3 = numpy.array ([4,4,5,6,7]) * q.s**-1
        self.assertEqual ( str(temp1 * temp3),  "[ 12.  16.  25.  36.  49.] (J) / s")
       
    def test_division(self):
        #arbitrary test of division
        self.assertAlmostEqual(  (10.3 * q.kPa) / (1 * q.inch) , 10.3 * q.kPa/q.inch)
        
        self.assertAlmostEqual ( (5.2 * q.eV) / (400.0 * q.eV) ,   q.Quantity(.013))
    
        # test division with compound units 
        self.assertAlmostEqual ( (5.2 * q.eV_) / (400.0 * q.eV_) ,   q.Quantity(.013))
        # the formatting should be the same 
        self.assertEqual (str( (5.2 * q.eV_)    / (400.0 * q.eV_)    ), str(  q.Quantity(.013)   ))
        
        
        
        #does division work with arrays?
        #divide an array with a scalar
        
        temp1  = numpy.array ([3,4,5,6,7]) * q.J_
        temp2 = .5 * q.s**-1
        
        self.assertEqual( str ( temp1 / temp2) , "[  6.   8.  10.  12.  14.] s * (J)")
        
        # divide an array with an array
        
        temp3 = numpy.array ([4,4,5,6,7]) * q.s**-1
        self.assertEqual ( str(temp1 / temp3),  "[ 0.75  1.    1.    1.    1.  ] s * (J)")
        
        

    def test_powering(self):  
        
        #test raising a quantity to a power
        
        self.assertAlmostEqual( (5.5 * q.cm) ** 5 , (5.5**5) * (q.cm**5))
        self.assertEqual( str (    (5.5 * q.cm) ** 5    ), str (     (5.5**5) * (q.cm**5)     )   )
        
        # must also work with compound units
        self.assertAlmostEqual ((5.5 * q.J_) ** 5 , (5.5**5) * (q.J_**5))
        self.assertEqual( str (    (5.5 * q.J_) ** 5    ), str(     (5.5**5) * (q.J_**5)     )    )
        
        # does powering work with arrays?
        temp = numpy.array([1, 2, 3,4,5]) * q.kg
        temp2 = (numpy.array([1,2,3,4,5]) **2) * q.kg**3
        
        self.assertEqual( str(temp**3 ), "[   1.    8.   27.   64.  125.] kg^3" )
        self.assertEqual(temp, temp2)
        self.assertEqual(str(temp), str (temp2))
        

    def test_getitem(self):
        
        tempArray1 = q.Quantity(numpy.array([1.5, 2.5 , 3, 5]), q.J_)
        temp = 2.5 * q.J_
        # check to see if quantites brought back from an array are good
        self.assertEqual(tempArray1[1], temp )
        #check the formatting
        self.assertEqual(str( tempArray1[1]  ) , str(temp)    )
        
        
        def tempfunc(index):
            return tempArray1[index]
        
        #make sure indexing is correct
        self.assertRaises(IndexError, tempfunc, 10)
        
    def test_setitem (self):
        temp = q.Quantity([0,2,5,7.6], q.lb)
        
        # needs to check for incompatible units
        def test(value):
            temp[2] = value
        
        #self.assertRaises(IncompatibleUnits, test, 60 * q.inch * q.J)
        # even in the case when the quantity has no units (maybe this could go away)
        #self.assertRaises(IncompatibleUnits, test, 60)
        
        #needs to check for out of bounds
        def tempfunc(value):
            temp[10] = value 
        
        self.assertRaises(IndexError, tempfunc, 5 * q.lb)
        
        
    def test_iterator(self):
        f = numpy.array ([100,200, 1,60 , -80])
        x = f * q.kPa_
        

        # make sure the iterator objects have the correct units
        for i in x:
            # currently fails
            self.assertEqual( i.units, q.kPa_.units)
        
#    def test_to_function(self):
#        temp = 100 * dyne * cm
#        
#        temp2 = temp.to( N_ * ft_)
#        
#        
#        self.assertAlmostEqual(temp, temp2)
#        
#        # need to find the actual conversion 
#        self.assertEqual ( str(temp2), " .     (N)/(ft)" )
        
    
    def test_specific_bugs(self):
        
        # bug 1) where trying to modify units to incompatible ones 
        temp = q.Quantity(1, q.lb)
        
        # needs to check for incompatible units
        def test(units):
            temp.units = units
        
        #this currently screws up q.J
        self.assertRaises(IncompatibleUnits, test, q.inch * q.J)

        #check for this bug
        self.assertEqual(str(q.J),   "Quantity(1.0), kg * m^2 / s^2")
        
        
        
        #bug 2)
        
        
        
        pass
    
    
    
 