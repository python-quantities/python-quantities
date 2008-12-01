"""
"""

from quantities.units.unitquantities import compound
from quantities.units.time import s, min

Hz = hertz = rps = s**-1
kHz = Hz * 1000
MHz = kHz * 1000
GHz = MHz * 1000
rpm = min**-1

Hz_ = hertz_ = rps_ = compound('Hz')
kHz_ = compound('kHz')
MHz_ = compound('MHz')
GHz_ = compound('GHz')
rpm_ = compound('rpm')

del compound, s, min
