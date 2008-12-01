"""
"""

from quantities.units.unitquantities import compound
from quantities.units.time import s
from quantities.units.mass import kg
from quantities.units.energy import J
from quantities.units.electromagnetism import coulomb

Bq = becquerel = becquerels = 1/s
Ci = curie = curies = 3.7e10 * becquerel
Gy = gray = grays = Sv = sievert = sieverts = J/kg
rem = rems = 1e-2 * sievert
rd = rad = rads = 1e-2 * gray
R = roentgen = roentgens = 2.58e-4 * coulomb/kg


Bq = becquerel = becquerels = compound('Bq')
Ci = curie = curies = compound('Ci')
Gy = gray = grays = Sv = sievert = sieverts = compound('Gy')
rem = rems = compound('rem')
rd = rad = rads = compound('rd')
R = roentgen = roentgens = compound('R')

del compound, s, kg, J, coulomb
