"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.length import m, nautical_mile
from quantities.units.time import s, h


c = \
    UnitQuantity('c', 299792458*m/s)
kt = knot = knots = knot_international = international_knot = \
    UnitQuantity('kt', nautical_mile/h)

del UnitQuantity, m, nautical_mile, s, h
