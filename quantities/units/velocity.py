"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.length import m, nmi
from quantities.units.time import s, h


c = \
    UnitQuantity('c', 299792458*m/s)
kt = knot = knots = knot_international = international_knot = \
    UnitQuantity('kt', nmi/h)

del UnitQuantity, m, nmi, s, h
