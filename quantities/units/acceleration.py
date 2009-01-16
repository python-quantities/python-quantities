# -*- coding: utf-8 -*-
"""
"""

from quantities.unitquantity import UnitQuantity
from quantities.units.time import s
from quantities.units.length import m

g_0 = g_n = gravity = standard_gravity = gee = force = free_fall = \
    standard_free_fall = gp = dynamic = geopotential = UnitQuantity(
    'standard_gravity',
    9.806650*m/s**2,
    symbol='g',
    u_symbol='g₀'
) # exact

del m, s, UnitQuantity
