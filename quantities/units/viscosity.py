"""
"""

from quantities.unitquantity import UnitQuantity
from quantities.units.time import s
from quantities.units.length import m
from quantities.units.pressure import Pa


P = poise = UnitQuantity(
    'poise',
    1e-1*Pa*s,
    symbol='P'
)
St = stokes = UnitQuantity(
    'stokes',
    1e-4*m**2/s,
    symbol='St'
)
rhe = UnitQuantity(
    'rhe',
    10/(Pa*s)
)

del UnitQuantity, s, m, Pa
