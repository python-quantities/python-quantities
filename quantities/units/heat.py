"""
"""

from quantities.units.unitquantities import compound
from quantities.units.temperature import K
from quantities.units.length import m
from quantities.units.power import W

clo = clos = 1.55e-1 * K * m**2 / W
clo_ = clos_ = compound('clo')

del compound, K, m, W
