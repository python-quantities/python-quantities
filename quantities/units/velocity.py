"""
"""

from quantities.units.unitquantities import compound
from quantities.units.length import m, nautical_mile
from quantities.units.time import s, h

c = 2.997925e+8 * m / s
kt = knot = knots = knot_international = international_knot = nautical_mile / h

c = compound('c')
kt = knot = knots = knot_international = international_knot = compound('kt')

del compound, m, nautical_mile, s, h
