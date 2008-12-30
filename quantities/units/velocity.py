"""
"""

from quantities.units.unitquantities import UnitQuantity
from quantities.units.length import m, nautical_mile
from quantities.units.time import s, h


c = UnitQuantity('c', 2.997925e+8*m/s)
kt = knot = knots = knot_international = international_knot = UnitQuantity('kt', nautical_mile/h)

del UnitQuantity, m, nautical_mile, s, h
