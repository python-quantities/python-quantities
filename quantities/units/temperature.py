"""
"""

from quantities.units.unitquantity import UnitTemperature

K = degK = kelvin = Kelvin = \
    UnitTemperature('K')
degR = rankine = Rankine = \
    UnitTemperature('Rankine', K/1.8)
degC = celsius = Celsius = \
    UnitTemperature('degC', K)
degF = fahrenheit = Fahrenheit = \
    UnitTemperature('degF', degR)
