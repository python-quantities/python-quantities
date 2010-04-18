"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .angle import revolution
from .time import s, min
from .dimensionless import count


Hz = hertz = rps = UnitQuantity(
    'hertz',
    s**-1,
    symbol='Hz'
)
kHz = kilohertz = UnitQuantity(
    'kilohertz',
    Hz*1000,
    symbol='kHz'
)
MHz = megahertz = UnitQuantity(
    'megahertz',
    kHz*1000,
    symbol='MHz'
)
GHz = gigahertz = UnitQuantity(
    'gigahertz',
    MHz*1000,
    symbol='GHz'
)
rpm = revolutions_per_minute = UnitQuantity(
    'revolutions_per_minute',
    revolution/min,
    symbol='rpm'
)
cps = UnitQuantity(
    'counts_per_second',
    count/s
)

del UnitQuantity, s, min
