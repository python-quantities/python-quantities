"""
"""

from quantities.units.unitquantities import compound
from quantities.units.acceleration import gravity
from quantities.units.mass import kg, pound
from quantities.units.length import m, mm, cm, inch, foot
from quantities.units.force import N, kip

Hg = hg = mercury = conventional_mercury = gravity * 13595.10 * kg / m**3
mercury_0C = mercury_32F = gravity * 13595.1 * kg / m**3
mercury_60F = gravity * 13556.8 * kg / m**3
H2O = h2o = water = conventional_water = gravity * 1000 * kg / m**3
water_4C = water_39F = gravity * 999.972 * kg / m**3
water_60F = gravity * 999.001 * kg / m**3

Hg = hg = mercury = conventional_mercury = compound('Hg')
mercury_0C = mercury_32F = compound('mercury_0C')
mercury_60F = compound('mercury_60F')
H2O = h2o = water = conventional_water = compound('H2O')
water_4C = water_39F = compound('water_4C')
water_60F = compound('water_60F')

Pa = pascal = N/m**2
kPa = kilopascal = kilopascals = 1000 * Pa
MPa = megapascal = megapascals = 1000 * kPa
GPa = gigapascal = gigapascals = 1000 * MPa
bar = bars = 1e5 * pascal
kbar = kilobar = kilobars = 1000 * bar
Mbar = kilobar = kilobars = 1000 * kbar
Gbar = gigabar = gigabars = 1000 * Mbar
atm = atmosphere = atmospheres = standard_atmosphere =standard_atmospheres = 1.01325e5 * pascal
at = technical_atmosphere = technical_atmospheres = 1 * kg * gravity/cm**2
inch_H2O_39F = inch * water_39F
inch_H2O_60F = inch * water_60F
inch_Hg_32F = inch * mercury_32F
inch_Hg_60F = inch * mercury_60F
millimeter_Hg_0C = mm * mercury_0C
footH2O = foot * water
cmHg = cm * Hg
cmH2O = cm * water
inHg = in_Hg = inch_Hg = inch * Hg
torr = torrs = mmHg = mm_Hg = millimeter_Hg = mm * Hg
foot_H2O = ftH2O = foot * water
psi = pound * gravity / inch**2
ksi = kip / inch**2
barie = baries = barye = baryes = 0.1 * N / m**2

Pa_ = pascal_ = compound('Pa')
kPa_ = kilopascal_ = kilopascals_ = compound('kPa')
MPa_ = megapascal_ = megapascals_ = compound('MPa')
GPa_ = gigapascal_ = gigapascals_ = compound('GPa')
bar_ = bars_ = compound('bar')
kbar_ = kilobar_ = kilobars_ = compound('kbar')
Mbar_ = kilobar_ = kilobars_ = compound('Mbar')
Gbar_ = gigabar_ = gigabars_ = compound('Gbar')
atm_ = atmosphere_ = atmospheres_ = standard_atmosphere_ =standard_atmospheres_ = compound('atm')
at_ = technical_atmosphere_ = technical_atmospheres_ = compound('at')
inch_H2O_39F_ = compound('inch_H2O_39F')
inch_H2O_60F_ = compound('inch_H2O_60F')
inch_Hg_32F_ = compound('inch_Hg_32F')
inch_Hg_60F_ = compound('inch_Hg_60F')
millimeter_Hg_0C_ = compound('millimeter_Hg_0C')
footH2O_ = compound('footH2O')
cmHg_ = compound('cmHg')
cmH2O_ = compound('cmH2O')
inHg_ = in_Hg_ = inch_Hg_ = compound('inHg')
torr_ = torrs_ = mmHg_ = mm_Hg_ = millimeter_Hg_ = compound('torr')
foot_H2O_ = ftH2O_ = compound('foot_H2O')
psi_ = compound('psi')
ksi_ = compound('ksi')
barie_ = baries_ = barye_ = baryes_ = compound('barie')

del compound, gravity, kg, pound, m, mm, cm, inch, foot, N, kip
