"""
"""

from quantities.units.unitquantities import compound
from quantities.units.time import s
from quantities.units.length import m
from quantities.units.pressure import Pa

poise = 1e-1 * Pa * s
St = stokes = 1e-4 * m**2/s
rhe = 10/(Pa * s)

poise = compound('poise')
St = stokes = compound('St')
rhe = compound('rhe')

del compound, s, m, Pa
