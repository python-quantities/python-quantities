"""
"""

from quantities.units.unitquantities import compound
from quantities.units.time import s
from quantities.units.length import m

g = force = gravity = free_fall = standard_free_fall = 9.806650 * m/s**2
gp = dynamic = geopotential = gravity

g_ = force_ = gravity_ = free_fall_ = standard_free_fall_ = compound('g')
gp_ = dynamic_ = geopotential_ = compound('gp')

del m, s, compound
