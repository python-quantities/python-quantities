"""
"""

from quantities.units.unitquantities import compound, information
from quantities.units.time import s

bit = bits = information('bit')
count = counts = information('counts')

Bd = baud = 1 / s
bps = bit / s
cps = 1 / s

Bd = baud = compound('Bd')
bps = compound('bps')
cps = compound('cps')

del compound, s
