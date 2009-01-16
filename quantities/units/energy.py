"""
"""

from quantities.unitquantity import UnitQuantity
from quantities.units.force import dyne, N
from quantities.units.length import cm, m
from quantities.units.time import s, h

J = joule = UnitQuantity(
    'joule',
    N*m,
    symbol='J',
    aliases=['joules']
)
erg = UnitQuantity(
    'erg',
    dyne*cm
)
btu = Btu = BTU = british_thermal_unit = UnitQuantity(
    'British_thermal_unit',
    J*1.05505585262e3,
    symbol='BTU'
)
eV = electron_volt = UnitQuantity(
    'electron_volt',
    J*1.60217653e-19,
    symbol='eV',
    aliases=['electron_volts']
) # TODO: move to constants
meV = UnitQuantity(
    'meV',
    eV/1000
)
keV = UnitQuantity(
    'keV',
    1000*eV
)
MeV = UnitQuantity(
    'MeV',
    1000*keV
)
bev = GeV = UnitQuantity(
    'GeV',
    1000*MeV
)
thm = therm = EC_therm = UnitQuantity(
    'EC_therm',
    100000*BTU,
    symbol='thm'
)
thermochemical_calorie = UnitQuantity(
    'thermochemical_calorie',
    4.184*J
)
cal = calorie = IT_calorie = UnitQuantity(
    'calorie',
    J*4.1868,
    symbol='cal',
    aliases=['calories', 'IT_calorie', 'IT_calories']
)
ton_TNT = UnitQuantity(
    'ton_TNT',
    4.184e9*J,
    symbol='tTNT'
)
US_therm = UnitQuantity(
    'US_therm',
    1.054804e8*J,
    aliases=['US_therms']
)
Wh = watthour = watt_hour = UnitQuantity(
    'watt_hour',
    J/s*h,
    symbol='Wh',
    aliases=['watthour', 'watthours', 'watt_hours']
)
kWh = kilowatthour = kilowatt_hour = UnitQuantity(
    'kilowatt_hour',
    1000*Wh,
    symbol='kWh',
    aliases=['kilowatthour', 'kilowatthours', 'kilowatt_hours']
)
MWh = megawatthour = megawatt_hour = UnitQuantity(
    'megawatt_hour',
    1000*kWh,
    symbol='MWh',
    aliases=['megawatthour', 'megawatthours', 'megawatt_hours']
)
GWh = gigawatthour = gigawatt_hour = UnitQuantity(
    'gigawatt_hour',
    1000*MWh,
    symbol='GWh',
    aliases=['gigawatthour', 'gigawatthours', 'gigawatt_hours']
)
E_h = hartree = hartree_energy = UnitQuantity(
    'hartree',
    4.35974394e-18*J,
    symbol='E_h',
    aliases=['hartrees', 'hartree_energy', 'Hartree_energy'],
    note='relative uncertainty = 2.1e-6'
) # TODO: move to physical constants

del UnitQuantity, dyne, N, cm, m, s, h
