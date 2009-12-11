# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .time import s
from .length import m

g_0 = g_n = gravity = standard_gravity = gee = force = free_fall = \
    standard_free_fall = gp = dynamic = geopotential = UnitQuantity(
    'standard_gravity',
    9.806650*m/s**2,
    symbol='g_0',
    u_symbol='g₀',
    doc='exact'
)

del m, s, UnitQuantity
