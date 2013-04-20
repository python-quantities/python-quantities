"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .length import m, nmi, cm, km
from .time import s, h


c = speed_of_light = UnitQuantity(
    'speed_of_light',
    299792458*m/s,
    symbol='c',
    doc='exact'
)
kt = knot = knot_international = international_knot = UnitQuantity(
    'nautical_miles_per_hour',
    nmi/h,
    symbol='kt',
    aliases=['knot', 'knots', 'knot_international', 'international_knot']
)

metre_per_second = UnitQuantity(
    'meter_per_second',
    m/s,
    aliases=['metres per second']
)
centimetre_per_second = UnitQuantity(
    'centimetre_per_second',
    cm/s,
    aliases=['centimetres_per_second']
)
kilometre_per_second = UnitQuantity(
    'kilometre_per_second',
    km/s,
    aliases=['kilometres_per_second']
)
kilometre_per_hour = UnitQuantity(
    'kilometre_per_hour',
    km/h,
    aliases=['kilometres_per_hour']
)

del UnitQuantity, m, nmi, s, h
