"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.time import s
from quantities.units.mass import kg
from quantities.units.energy import J
from quantities.units.electromagnetism import coulomb


Bq = becquerel = becquerels = \
    UnitQuantity('Bq', 1/s)
Ci = curie = curies = \
    UnitQuantity('Ci', 3.7e10*becquerel)
Gy = gray = grays = Sv = sievert = sieverts = \
    UnitQuantity('Gy', J/kg)
rem = rems = \
    UnitQuantity('rem', 1e-2*sievert)
rd = rad = rads = \
    UnitQuantity('rd', 1e-2*gray)
R = roentgen = roentgens = \
    UnitQuantity('R', 2.58e-4*coulomb/kg)

del UnitQuantity, s, kg, J, coulomb
