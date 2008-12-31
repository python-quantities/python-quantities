"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.length import m, rod

are = ares = UnitQuantity('are', 100*m**2)
b = barn = barns_ = UnitQuantity('b', 1e-28*m**2)
circular_mil = UnitQuantity('circular_mil', 5.067075e-10*m**2)
darcy = UnitQuantity('darcy', 9.869233e-13*m**2)
hectare = UnitQuantity('hectare', 1e4*m**2)
acre = acres = UnitQuantity('acre', 160*rod**2)

kayser = UnitQuantity('kayser', 1e2/m)

del UnitQuantity, m, rod
