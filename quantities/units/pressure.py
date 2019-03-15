"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .acceleration import gravity
from .mass import g, kg, pound
from .length import m, mm, cm, inch, ft
from .force import N, kip


Hg = mercury = conventional_mercury = UnitQuantity(
    'conventional_mercury',
    gravity*13.59510*g/cm**3
)
mercury_60F = UnitQuantity('mercury_60F', gravity*13556.8*kg/m**3)
H2O = h2o = water = conventional_water = UnitQuantity('H2O', gravity*1000*kg/m**3)
water_4C = water_39F = UnitQuantity('water_4C', gravity*999.972*kg/m**3)
water_60F = UnitQuantity('water_60F', gravity*999.001*kg/m**3)

Pa = pascal = UnitQuantity(
    'pascal',
    N/m**2,
    symbol='Pa',
    aliases=['pascals']
)
hPa = hectopascal = UnitQuantity(
    'hectopascal',
    100*Pa,
    symbol='hPa',
)
kPa = kilopascal = UnitQuantity(
    'kilopascal',
    1000*Pa,
    symbol='kPa',
    aliases=['kilopascals']
)
MPa = megapascal = UnitQuantity(
    'megapascal',
    1000*kPa,
    symbol='MPa',
    aliases=['megapascals']
)
GPa = gigapascal = UnitQuantity(
    'gigapascal',
    1000*MPa,
    symbol='GPa',
    aliases=['gigapascals']
)
bar = UnitQuantity(
    'bar',
    100000*pascal,
    aliases=['bars']
)
mb = mbar = millibar = UnitQuantity(
    'millibar',
    0.001*bar,
    symbol='mb',
    aliases=['mbar']
)
kbar = kilobar = UnitQuantity(
    'kilobar',
    1000*bar,
    symbol='kbar',
    aliases=['kilobars']
)
Mbar = megabar = UnitQuantity(
    'megabar',
    1000*kbar,
    symbol='Mbar',
    aliases=['megabars']
)
Gbar = gigabar = UnitQuantity(
    'gigabar',
    1000*Mbar,
    symbol='Gbar',
    aliases=['gigabars']
)
atm = atmosphere = standard_atmosphere = UnitQuantity(
    'standard_atmosphere',
    101325*pascal,
    symbol='atm',
    aliases=['atmosphere', 'atmospheres', 'standard_atmospheres']
)
at = technical_atmosphere = UnitQuantity(
    'technical_atmosphere',
    kg*gravity/cm**2,
    symbol='at',
    aliases=['technical_atmospheres']
)
torr = UnitQuantity(
    'torr',
    atm/760
)
psi = pound_force_per_square_inch = UnitQuantity(
    'pound_force_per_square_inch',
    pound*gravity/inch**2,
    symbol='psi'
)
ksi = kip_per_square_inch = UnitQuantity(
    'kip_per_square_inch',
    kip/inch**2,
    symbol='ksi'
)
barye = barie = barad = barad = barrie = baryd = UnitQuantity(
    'barye',
    0.1*N/m**2,
    symbol='Ba',
    aliases=[
        'barie', 'baries', 'baryes', 'barad', 'barads', 'barrie', 'baryd',
        'baryed'
    ]
)

mmHg = mm_Hg = millimeter_Hg = millimeter_Hg_0C = UnitQuantity(
    'millimeter_Hg',
    mm*mercury,
    symbol='mmHg',
    aliases=['mm_Hg', 'millimeter_Hg_0C'],
    doc="""
    The pressure exerted at the base of a column of fluid exactly 1 mm high,
    when the density of the fluid is exactly 13.5951 g/cm^3, at a place where
    the acceleration of gravity is exactly 9.80665 m/s^2.

    http://en.wikipedia.org/wiki/Conventional_millimeter_of_mercury
    """
)
cmHg = cm_Hg = centimeter_Hg = UnitQuantity(
    'cmHg',
    cm*Hg,
    aliases=['cm_Hg', 'centimeter_Hg']
)
inHg = in_Hg = inch_Hg = inch_Hg_32F = UnitQuantity(
    'inHg',
    inch*Hg,
    aliases=['in_Hg', 'inch_Hg', 'inch_Hg_32F']
)
inch_Hg_60F = UnitQuantity(
    'inch_Hg_60F',
    inch*mercury_60F
)

inch_H2O_39F = UnitQuantity(
    'inch_H2O_39F',
    inch*water_39F
)
inch_H2O_60F = UnitQuantity(
    'inch_H2O_60F',
    inch*water_60F
)
footH2O = UnitQuantity(
    'footH2O',
    ft*water
)
cmH2O = UnitQuantity(
    'cmH2O',
    cm*water
)
foot_H2O = ftH2O = UnitQuantity(
    'foot_H2O',
    ft*water,
    aliases=['ftH2O']
)

del UnitQuantity, gravity, kg, pound, m, mm, cm, inch, ft, N, kip
