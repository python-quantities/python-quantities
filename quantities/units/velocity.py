"""
"""

from quantities.unitquantity import UnitQuantity
from quantities.units.length import m, nmi
from quantities.units.time import s, h


c = speed_of_light = UnitQuantity(
    'speed_of_light',
    299792458*m/s,
    symbol='c'
)
kt = knot = knot_international = international_knot = UnitQuantity(
    'nautical_miles_per_hour',
    nmi/h,
    symbol='kt',
    aliases=['knot', 'knots', 'knot_international', 'international_knot']
)

del UnitQuantity, m, nmi, s, h
