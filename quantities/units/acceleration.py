"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.time import s
from quantities.units.length import m

g = gravity = force = free_fall = standard_free_fall = gp = dynamic = \
    geopotential = \
    UnitQuantity('g', 9.806650*m/s**2) # TODO: check

del m, s, UnitQuantity
