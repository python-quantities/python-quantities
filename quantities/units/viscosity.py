"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.time import s
from quantities.units.length import m
from quantities.units.pressure import Pa


poise = \
    UnitQuantity('poise', 1e-1*Pa*s)
St = stokes = \
    UnitQuantity('St', 1e-4*m**2/s)
rhe = \
    UnitQuantity('rhe', 10/(Pa*s))

del UnitQuantity, s, m, Pa
