"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .temperature import K, degF
from .length import m, ft
from .power import W
from .energy import BTU
from .time import h


RSI = UnitQuantity(
    'RSI',
    K*m**2/W,
    doc='R-value in SI'
)

clo = clos = UnitQuantity(
    'clo',
    0.155*RSI,
    aliases=['clos']
)

R_value = UnitQuantity(
    'R_value',
    ft**2*degF*h/BTU,
    doc='American customary units'
)


del UnitQuantity, K, degF, m, ft, W, BTU, h
