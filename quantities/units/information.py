"""
"""

from quantities.units.unitquantity import UnitQuantity, UnitInformation
from quantities.units.time import s

bit = bits = \
    UnitInformation('bit')
count = counts = \
    UnitInformation('counts')
Bd = baud = \
    UnitQuantity('Bd', 1/s)
bps = \
    UnitQuantity('bps', bit/s)
cps = \
    UnitQuantity('cps', 1/s)

del UnitQuantity, s
