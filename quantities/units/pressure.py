"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.acceleration import gravity
from quantities.units.mass import kg, pound
from quantities.units.length import m, mm, cm, inch, foot
from quantities.units.force import N, kip


Hg = hg = mercury = conventional_mercury = \
    UnitQuantity('Hg', gravity*13595.10*kg/m**3)
mercury_0C = mercury_32F = \
    UnitQuantity('mercury_0C', gravity*13595.1*kg/m**3)
mercury_60F = \
    UnitQuantity('mercury_60F', gravity*13556.8*kg/m**3)
H2O = h2o = water = conventional_water = \
    UnitQuantity('H2O', gravity*1000*kg/m**3)
water_4C = water_39F = \
    UnitQuantity('water_4C', gravity*999.972*kg/m**3)
water_60F = \
    UnitQuantity('water_60F', gravity*999.001*kg/m**3)

Pa = pascal = \
    UnitQuantity('Pa', N/m**2)
kPa = kilopascal = kilopascals = \
    UnitQuantity('kPa', 1000*Pa)
MPa = megapascal = megapascals = \
    UnitQuantity('MPa', 1000*kPa)
GPa = gigapascal = gigapascals = \
    UnitQuantity('GPa', 1000*MPa)
bar = bars = UnitQuantity('bar', 1e5*pascal)
kbar = kilobar = kilobars = \
    UnitQuantity('kbar', 1000*bar)
Mbar = kilobar = kilobars = \
    UnitQuantity('Mbar', 1000*kbar)
Gbar = gigabar = gigabars = \
    UnitQuantity('Gbar', 1000*Mbar)
atm = atmosphere = atmospheres = standard_atmosphere =standard_atmospheres = \
    UnitQuantity('atm', 1.01325e5*pascal)
at = technical_atmosphere = technical_atmospheres = \
    UnitQuantity('at', kg*gravity/cm**2)
inch_H2O_39F = \
    UnitQuantity('inch_H2O_39F', inch*water_39F)
inch_H2O_60F = \
    UnitQuantity('inch_H2O_60F', inch*water_60F)
inch_Hg_32F = \
    UnitQuantity('inch_Hg_32F', inch*mercury_32F)
inch_Hg_60F = \
    UnitQuantity('inch_Hg_60F', inch*mercury_60F)
millimeter_Hg_0C = \
    UnitQuantity('millimeter_Hg_0C', mm*mercury_0C)
footH2O = \
    UnitQuantity('footH2O', foot*water)
cmHg = \
    UnitQuantity('cmHg', cm*Hg)
cmH2O = \
    UnitQuantity('cmH2O', cm*water)
inHg = in_Hg = inch_Hg = \
    UnitQuantity('inHg', inch*Hg)
torr = torrs = mmHg = mm_Hg = millimeter_Hg = \
    UnitQuantity('torr', mm*Hg)
foot_H2O = ftH2O = \
    UnitQuantity('foot_H2O', foot*water)
psi = \
    UnitQuantity('psi', pound*gravity/inch**2)
ksi = \
    UnitQuantity('ksi', kip/inch**2)
barie = baries = barye = baryes = \
    UnitQuantity('barie', 0.1*N/m**2)

del UnitQuantity, gravity, kg, pound, m, mm, cm, inch, foot, N, kip
