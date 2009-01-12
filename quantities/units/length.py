"""
"""

from quantities.units.unitquantity import UnitLength

m = UnitLength(
    'm',
    aliases=['meter', 'meters', 'metre', 'metres']
)
km = UnitLength(
    'km', 1000*m, aliases=['kilometer', 'kilometers', 'kilometre', 'kilometres']
)
cm = UnitLength(
    'cm', m/100,
    aliases=['centimeter', 'centimeters', 'centimetre', 'centimetres']
)
mm = UnitLength(
    'mm', m/1000,
    aliases=['millimeter', 'millimeters', 'millimetre', 'millimetres']
)
um = UnitLength(
    'um', mm/1000,
    aliases=[
        'micron', 'microns', 'micrometer', 'micrometers', 'micrometre',
        'micrometres'
    ]
)
nm = UnitLength(
    'nm', um/1000,
    aliases=['nanometer', 'nanometers', 'nanometre', 'nanometres']
)
pm = UnitLength(
    'pm', nm/1000,
    aliases=['picometer', 'picometers', 'picometre', 'picometres']
)
angstrom = UnitLength(
    'angstroms', nm/10,
    aliases=['angstroms']
)
fm = UnitLength(
    'fm', nm/1000,
    aliases=['femtometer', 'femtometer', 'femtometre', 'femtometres']
)

inch = UnitLength(
    'inch', 2.54*cm,
    aliases=['inches', 'international_inch', 'international_inches']
)
ft = UnitLength(
    'ft', 12*inch,
    aliases=['foot', 'feet', 'international_foot', 'international_feet']
)
mi = UnitLength(
    'mi', 5280*ft,
    aliases=['mile', 'miles', 'international_mile', 'international_miles']
)
yd = UnitLength(
    'yd', 3*ft,
    aliases=['yard', 'yards', 'international_yard', 'international_yards']
)
mil = UnitLength(
    'mil', 2.54e-5*m,
    aliases=['mils']
)
pc = UnitLength(
    'pc', 3.08568025e16*m,
    aliases=['parsec', 'parsecs']
)
ly = UnitLength(
    'light_year', 9.4605284e15*m,
    aliases=['light_year', 'light_years']
)
au = UnitLength(
    'au', 149597870691*m,
    aliases=['astronomical_unit', 'astronomical_units']
)
fermi = UnitLength(
    'fermi', 1e-15*m,
    aliases=['fermis']
)
nmi = UnitLength(
    'nautical_mile', 1.852e3*m,
    aliases=['nmile', 'nmiles', 'nautical_mile', 'nautical_miles']
)
printers_point = UnitLength(
    'printers_point', 3.514598e-4*m, # TODO: check
    aliases=['printers_points']
)
pica = UnitLength(
    'pica', 12*printers_point,
    aliases=['picas', 'printers_pica', 'printers_picas']
)

US_survey_foot = UnitLength(
    'US_survey_foot', 1200*m/3937,
    aliases=['US_survey_feet']
)
US_survey_yard = UnitLength(
    'US_survey_yard', 3*US_survey_foot,
    aliases=['US_survey_yards']
)
US_survey_mile = UnitLength(
    'US_survey_mile', 5280*US_survey_foot,
    aliases=['US_survey_miles']
)
US_statute_mile = UnitLength(
    'US_statute_mile', US_survey_mile,
    aliases=['US_statute_miles']
)
rod = UnitLength(
    'rod', 16.5*US_survey_foot, # TODO: check NIST, google uses foot
    aliases=['rods', 'pole', 'poles', 'perch', 'perches']
)
furlong = UnitLength(
    'furlong', 660*US_survey_foot,
    aliases=['furlongs']
)
fathom = UnitLength(
    'fathom', 6*US_survey_foot,
    aliases=['fathoms']
)
chain = UnitLength(
    'chain', 2.011684e1*m, # TODO: check
    aliases=['chains']
)
big_point = UnitLength(
    'big_point', inch/72,
    aliases=['big_points']
)
barleycorn = UnitLength(
    'barleycorn', inch/3,
    aliases=['barleycorns']
)
arpentlin = UnitLength(
    'arpentlin', 191.835*ft
)
