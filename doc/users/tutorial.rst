====================
Quick-start tutorial
====================

This is just a placeholder tutorial, it needs to be developed::

  >>> q=Quantity([1,2,3], 'J')

or::

  >>> q=Quantity([1,2,3], J)

or::

  >>> q = array([1,2,3]) * J
  >>> q
  [ 1.,  2.,  3.]*J

  >>> q.units = 'ft'
  ValueError: Cannot convert between quanitites with units of 'J' and 'ft'

  >>> q = 1*m
  >>> q.units = ft
  >>> q
  3.280839895013123*ft


here's some tricks for preserving compound units::

  >>> q=19*UnitQuantity("(parsec/cm**3)", parsec/cm**3)*cm**3
  >>> q
  19.0*cm**3*(parsec/cm**3)

which can be simplified::

  >>> qs = q.simplified
  >>> qs
  5.862792475e+17*m

and then rescaled back into compound units::

  >>> qs.rescale(UnitQuantity("(parsec/cm**3)", parsec/cm**3)*cm**3)
  >>> qs
  19.0*cm**3*(parsec/cm**3)
