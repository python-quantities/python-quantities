"""
"""

from quantities.unitquantity import UnitQuantity
from quantities.units.time import s, min


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
    min**-1,
    symbol='rpm'
)

del UnitQuantity, s, min
