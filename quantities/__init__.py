"""

>>> q=Quantity([1,2,3.0], 'J')

or

>>> q=Quantity([1,2,3.0], J)

or

>>> q = array([1,2,3.0]) * J
>>> q
Quantity([ 1.,  2.,  3.]), kg * m^2 / s^2

>>> q.units = 'ft'
IncompatibleUnits: Cannot convert between quanitites with units of 'kg m^2 s^-2' and 'ft'

>>> q = 1*m
>>> q.units = ft
>>> q
Quantity(3.280839895013123), ft


here's some tricks for preserving compound units:

>>> q=Quantity(19,'compound("parsec/cm^3")*J')

or:

>>> q=Quantity(19,'compound("parsec/cm^3")*compound("m^3/m^2")')

or:

>>> q=19*compound("parsec/cm^3")*compound("m^3/m^2")

and here is how to reduce compound units to their irreducible form:

>>> q.reduce_units()
>>> q
Quantity(5.8627881999999992e+23), 1 / m

>>> q.units=compound("parsec/cm^3")*compound("m^3/m^2")
>>> q
Quantity(19.000000000000004), (m^3/m^2) * (parsec/cm^3)


"""


__version__ = '0.2'


from quantity import Quantity, ProtectedUnitsError

from parser import unit_registry

import units
from units import *

import constants
from constants import *
