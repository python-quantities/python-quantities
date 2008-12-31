"""

>>> q=Quantity([1,2,3.0], 'J')

or

>>> q=Quantity([1,2,3.0], J)

or

>>> q = array([1,2,3.0]) * J
>>> q
[ 1.,  2.,  3.]*J

>>> q.units = 'ft'
TypeError: Cannot convert between quanitites with units of 'J' and 'ft'

>>> q = 1*m
>>> q.units = ft
>>> q
3.280839895013123*ft


here's some tricks for preserving compound units:

>>> q=19*UnitQuantity("(parsec/cm**3)")*J

or:

>>> q=19*UnitQuantity("(parsec/cm**3)"*UnitQuantity("(m**3/m**2)")

"""


__version__ = '0.1(bzr)'


from quantity import Quantity

from parser import unit_registry

import units
from units import *

import constants
from constants import *
