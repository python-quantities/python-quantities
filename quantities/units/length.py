# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitLength, UnitQuantity

m = meter = metre = UnitLength(
    'meter',
    symbol='m',
    aliases=['meters', 'metre', 'metres']
)
km = kilometer = kilometre = UnitLength(
    'kilometer',
    1000*m,
    symbol='km',
    aliases=['kilometers', 'kilometre', 'kilometres']
)
cm = centimeter = centimetre = UnitLength(
    'centimeter',
    m/100,
    'cm',
    aliases=['centimeters', 'centimetre', 'centimetres']
)
mm = millimeter = millimetre = UnitLength(
    'millimeter',
    m/1000,
    symbol='mm',
    aliases=['millimeters', 'millimetre', 'millimetres']
)
um = micrometer = micrometre = micron = UnitLength(
    'micrometer',
    mm/1000,
    symbol='um',
    u_symbol='µm',
    aliases=[
        'micron', 'microns', 'micrometers', 'micrometre', 'micrometres'
    ]
)
nm = nanometer = nanometre = UnitLength(
    'nanometer',
    um/1000,
    symbol='nm',
    aliases=['nanometers', 'nanometre', 'nanometres']
)
pm = picometer = picometre = UnitLength(
    'picometer',
    nm/1000,
    symbol='pm',
    aliases=['picometers', 'picometre', 'picometres']
)
angstrom = UnitLength(
    'angstrom',
    nm/10,
    u_symbol='Å',
    aliases=['angstroms']
)
fm = femtometer = femtometre = fermi = UnitLength(
    'femtometer',
    pm/1000,
    symbol='fm',
    aliases=['femtometers', 'femtometre', 'femtometres', 'fermi', 'fermis']
)

inch = international_inch = UnitLength(
    'inch',
    2.54*cm,
    symbol='in',
    aliases=['inches', 'international_inch', 'international_inches']
)
ft = foot = international_foot = UnitLength(
    'foot',
    12*inch,
    symbol='ft',
    aliases=['feet', 'international_foot' 'international_feet']
)
mi = mile = international_mile = UnitLength(
    'mile',
    5280*ft,
    symbol='mi',
    aliases=['miles', 'international_mile', 'international_miles']
)
yd = yard = international_yard = UnitLength(
    'yard',
    3*ft,
    symbol='yd',
    aliases=['yards', 'international_yard', 'international_yards']
)
mil = thou = UnitLength(
    'mil',
    inch/1000,
    aliases=['mils', 'thou', 'thous']
)
pc = parsec = UnitLength(
    'parsec',
    3.08568025e16*m,
    symbol='pc',
    aliases=['parsecs'],
    doc='approximate'
)
ly = light_year = UnitLength(
    'light_year',
    9460730472580.8*km,
    symbol='ly',
    aliases=['light_years']
)
au = astronomical_unit = UnitLength(
    'astronomical_unit',
    149597870691*m,
    symbol='au',
    aliases=['astronomical_units'],
    doc='''
    An astronomical unit (abbreviated as AU, au, a.u., or sometimes ua) is a
    unit of length roughly equal to the mean distance between the Earth and
    the Sun. It is approximately 150 million kilometres (93 million miles).

    uncertainty ± 30 m

    http://en.wikipedia.org/wiki/Astronomical_unit
    '''
)

nmi = nautical_mile = UnitLength(
    'nautical_mile',
    1.852e3*m,
    symbol='nmi',
    aliases=['nmile', 'nmiles', 'nautical_miles']
)
pt = printers_point = point = UnitLength(
    'printers_point',
    127*mm/360,
    symbol='point',
    aliases=['printers_points', 'points'],
    doc='pt is reserved for pint'
)
pica = UnitLength(
    'pica',
    12*printers_point,
    aliases=['picas', 'printers_pica', 'printers_picas']
)

US_survey_foot = UnitLength(
    'US_survey_foot',
    1200*m/3937,
    aliases=['US_survey_feet']
)
US_survey_yard = UnitLength(
    'US_survey_yard',
    3*US_survey_foot,
    aliases=['US_survey_yards']
)
US_survey_mile = US_statute_mile = UnitLength(
    'US_survey_mile',
    5280*US_survey_foot,
    aliases=['US_survey_miles', 'US_statute_mile', 'US_statute_miles']
)
rod = pole = perch = UnitLength(
    'rod',
    16.5*US_survey_foot,
    aliases=['rods', 'pole', 'poles', 'perch', 'perches']
)
furlong = UnitLength(
    'furlong',
    660*US_survey_foot,
    aliases=['furlongs']
)
fathom = UnitLength(
    'fathom',
    6*US_survey_foot,
    aliases=['fathoms']
)
chain = UnitLength(
    'chain',
    66*US_survey_foot,
    aliases=['chains']
)
barleycorn = UnitLength(
    'barleycorn',
    inch/3,
    aliases=['barleycorns']
)
arpentlin = UnitLength(
    'arpentlin',
    191.835*ft
)

kayser = wavenumber = UnitQuantity(
    'kayser',
    1/cm,
    aliases=['kaysers', 'wavenumber', 'wavenumbers']
)

del UnitQuantity
