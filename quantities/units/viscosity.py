"""
"""

from ..unitquantity import UnitQuantity
from .length import m
from .pressure import Pa
from .time import s

P = poise = UnitQuantity(
    'poise',
    1e-1*Pa*s,
    symbol='P'
)
cP = centipoise = UnitQuantity(
    'centipoise',
    P/100,
    symbol='cP'
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
