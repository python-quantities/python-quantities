"""
"""

from quantities.units.unitquantities import compound
from quantities.units.force import dyne, N
from quantities.units.length import cm, m
from quantities.units.time import s, h

J = joule = joules = N*m
erg = dyne*cm
btu = Btu = BTU = btus = BTUs = IT_Btu = IT_Btus = J * 1.05505585262e3
eV = electron_volt = electron_volts = J * 1.602177e-19
meV = eV/1000
keV = 1000*eV
MeV = 1000*keV
bev = GeV = 1000 * MeV
EC_therm = EC_therms = 1.05506e8 * J
thermochemical_calorie = thermochemical_calories = 4.184000 * J
cal = calorie = calories = IT_calorie = IT_calories = J * 4.1868
ton_TNT = 4.184e9 * J
thm = therm = therms = US_therm = US_therms = 1.054804e8 * J
Wh = watthour = watthours = J/s*h
kWh = kilowatthour = kilowatthours = 1000 * Wh
MWh = megawatthour = megawatthours = 1000 * kWh
GWh = gigawatthour = gigawatthours = 1000 * MWh
E_h = Hartree_energy = 4.35974394e-18 * J

J_ = joule_ = joules_ = compound('J')
erg_ = compound('erg')
btu_ = Btu_ = BTU_ = btus_ = BTUs_ = IT_Btu_ = IT_Btus_ = compound('Btu')
eV_ = electron_volt_ = electron_volts_ = compound('eV')
meV_ = compound('meV')
keV_ = compound('keV')
MeV_ = compound('MeV')
bev_ = GeV_ = compound('GeV')
EC_therm_ = EC_therms_ = compound('EC_therm')
thermochemical_calorie_ = thermochemical_calories_ = compound('thermochemical_calorie')
cal_ = calorie_ = calories_ = IT_calorie_ = IT_calories_ = compound('cal')
ton_TNT_ = compound('ton_TNT')
thm_ = therm_ = therms_ = US_therm_ = US_therms_ = compound('thm')
Wh_ = watthour_ = watthours_ = compound('Wh')
kWh_ = kilowatthour_ = kilowatthours_ = compound('kWh')
MWh_ = megawatthour_ = megawatthours_ = compound('MWh')
GWh_ = gigawatthour_ = gigawatthours_ = compound('GWh')
E_h_ = Hartree_energy_ = compound('E_h')

del compound, dyne, N, cm, m, s, h
