"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.time import s, min


Hz = hertz = rps = \
    UnitQuantity('Hz', s**-1)
kHz = \
    UnitQuantity('kHz', Hz*1000)
MHz = \
    UnitQuantity('MHz', kHz*1000)
GHz = \
    UnitQuantity('GHz', MHz*1000)
rpm = \
    UnitQuantity('rpm', min**-1)

del UnitQuantity, s, min
