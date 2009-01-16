"""
"""

from quantities.unitquantity import UnitQuantity
from quantities.units.temperature import K, degF
from quantities.units.length import m, ft
from quantities.units.power import W
from quantities.units.energy import BTU
from quantities.units.time import h


RSI = UnitQuantity(
    'RSI',
    K*m**2/W,
    note='R-value in SI'
)

clo = clos = UnitQuantity(
    'clo',
    0.155*RSI,
    aliases=['clos']
)

R_value = UnitQuantity(
    'R_value',
    ft**2*degF*h/BTU,
    note='American customary units'
)


del UnitQuantity, K, degF, m, ft, W, BTU, h
