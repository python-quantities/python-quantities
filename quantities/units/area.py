"""
"""

from quantities.units.unitquantities import compound
from quantities.units.length import m, rod

are = ares = 100 * m**2
b = barn = barns = 1e-28 * m**2
circular_mil = 5.067075e-10 *m**2
darcy = 9.869233e-13 * m**2
hectare = 1e4 * m**2
acre = acres = 160 * rod**2

kayser = 1e2/m

are_ = ares_ = compound('are')
b_ = barn_ = barns_ = compound('b')
circular_mil_ = compound('circular_mil')
darcy_ = compound('darcy')
hectare_ = compound('hectare')
acre_ = acres_ = compound('acre')

kayser_ = compound('kayser')

del compound, m, rod
