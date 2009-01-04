"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.temperature import K
from quantities.units.length import m
from quantities.units.power import W


clo = clos = \
    UnitQuantity('clo', 1.55e-1*K*m**2/W)

del UnitQuantity, K, m, W
