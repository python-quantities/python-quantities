"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.time import s
from quantities.units.length import m

g = gravity = UnitQuantity('g', 9.806650*m/s**2)
force = free_fall = standard_free_fall = gp = dynamic = geopotential = g

del m, s, UnitQuantity
