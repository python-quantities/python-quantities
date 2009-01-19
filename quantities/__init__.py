"""

This is a crash-course introduction to the Quantities package::

  >>> from quantities import *
  >>> q = Quantity([1,2,3], 'J')

or::

  >>> q = Quantity([1,2,3], J)

or::

  >>> q = array([1,2,3]) * J
  >>> q
  [ 1.,  2.,  3.]*J

Units can be converted in many cases::

  >>> q = 1*m
  >>> q.units = ft
  >>> q
  3.280839895013123*ft

but will fail if the requested units fails a dimensional analysis::

  >>> q.rescale('J')
  ValueError: Cannot convert between quanitites with units of 'ft' and 'J'

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

There is also support for quantities with uncertainty::

  >>> q = UncertainQuantity(4,J,.2)
  >>> q
  4.0*J
  +/-0.2*J (1 sigma)

By assuming that the uncertainties are uncorrelated, the uncertainty can be
propagated during arithmetic operations::

  >>> length = UncertainQuantity(2.0,m,.001)
  >>> width = UncertainQuantity(3.0,m,.001)
  >>> area = length*width
  >>> area
  6.0*m**2
  +/-0.00360555127546*m**2 (1 sigma)

In that case, the measurements of the length and width were independent, and
the two uncertainties presumed to be uncorrelated. Here is a warning though:

  >>> q*q
  16.0*J**2
  +/-1.1313708499*J**2 (1 sigma)

This result is probably incorrect, since it assumes the uncertainties of the two
multiplicands are uncorrelated. It would be more accurate in this case to use::

  >>> q**2
  16.0*J**2
  +/-1.6*J**2 (1 sigma)
"""

from __future__ import absolute_import

__version__ = '0.1(bzr)'

from .quantity import Quantity
from .uncertainquantity import UncertainQuantity
from .unitquantity import *

from .units import *

from . import constants
