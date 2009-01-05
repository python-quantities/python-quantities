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


Here's some tricks for working compound units, which can be preserved::

  >>> q=19*CompoundUnit("parsec/cm**3")*cm**3
  >>> q
  19.0*cm**3*(parsec/cm**3)

and can be simplified::

  >>> qs = q.simplified
  >>> qs
  5.862792475e+17*m

and then rescaled back into compound units::

  >>> qs.rescale(CompoundUnit("parsec/cm**3")*cm**3)
  >>> qs
  19.0*cm**3*(parsec/cm**3)
"""


__version__ = '0.1(bzr)'

from quantity import Quantity, UncertainQuantity

from parser import unit_registry

import units
from units import *

import constants
from constants import *
