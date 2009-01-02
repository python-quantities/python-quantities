"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.force import dyne, N
from quantities.units.length import cm, m
from quantities.units.time import s, h

J = joule = joules = \
    UnitQuantity('J', N*m)
erg = \
    UnitQuantity('erg', dyne*cm)
btu = Btu = BTU = btus = BTUs = IT_Btu = IT_Btus = \
    UnitQuantity('Btu', J*1.05505585262e3)
eV = electron_volt = electron_volts = \
    UnitQuantity('eV', J*1.602177e-19)
meV = \
    UnitQuantity('meV', eV/1000)
keV = \
    UnitQuantity('keV', 1000*eV)
MeV = \
    UnitQuantity('MeV', 1000*keV)
bev = GeV = \
    UnitQuantity('GeV', 1000*MeV)
EC_therm = EC_therms = \
    UnitQuantity('EC_therm', 1.05506e8*J)
thermochemical_calorie = thermochemical_calories = \
    UnitQuantity('thermochemical_calorie', 4.184000*J)
cal = calorie = calories = IT_calorie = IT_calories = \
    UnitQuantity('cal', J*4.1868)
ton_TNT = \
    UnitQuantity('ton_TNT', 4.184e9*J)
thm = therm = therms = US_therm = US_therms = \
    UnitQuantity('thm', 1.054804e8*J)
Wh = watthour = watthours = \
    UnitQuantity('Wh', J/s*h)
kWh = kilowatthour = kilowatthours = \
    UnitQuantity('kWh', 1000*Wh)
MWh = megawatthour = megawatthours = \
    UnitQuantity('MWh', 1000*kWh)
GWh = gigawatthour = gigawatthours = \
    UnitQuantity('GWh', 1000*MWh)
E_h = Hartree_energy = \
    UnitQuantity('E_h', 4.35974394e-18*J)

del UnitQuantity, dyne, N, cm, m, s, h
