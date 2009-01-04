"""
"""

from quantities.units.unitquantity import UnitLength

m = meter = meters = metre = metres = \
    UnitLength('m')
km = kilometer = kilometers = kilometre = kilometres = \
    UnitLength('km', 1000*m)
cm = centimeter = centimeters = centimetre = centimetres = \
    UnitLength('cm', m/100)
mm = millimeter = millimeters = millimetre = millimetres = \
    UnitLength('mm', m/1000)
um = micron = microns = micrometer = micrometers = micrometre = micrometres = \
    UnitLength('um', mm/1000)
nm = nanometer = nanometers = nanometre = nanometres = \
    UnitLength('nm', um/1000)
pm = picometer = picometers = picometre = picometres = \
    UnitLength('pm', nm/1000)
angstrom = angstroms = \
    UnitLength('angstroms', nm/10)
fm = femtometer = femtometer = femtometre = femtometres = \
    UnitLength('fm', nm/1000)

inch = inches = international_inch = international_inches = \
    UnitLength('inch', 2.54*cm)
ft = foot = feet = international_foot = international_feet = \
    UnitLength('ft', 12*inch)
mi = mile = miles = international_mile = international_miles = \
    UnitLength('mi', 5280*ft)
yd = yard = yards = international_yard = international_yards = \
    UnitLength('yd', 3*ft)
mil = mils = \
    UnitLength('mil', 2.54e-5*m)
pc = parsec = parsecs = \
    UnitLength('parsec', 3.08568025e16*meter)
ly = light_year = light_years = \
    UnitLength('light_year', 9.4605284e15*m)
au = astronomical_unit = astronomical_units = \
    UnitLength('au', 149597870691*m)
fermi = fermis = \
    UnitLength('fermi', 1e-15*m)
nmi = nmile = nmiles = nautical_mile = nautical_miles = \
    UnitLength('nautical_mile', 1.852e3*m)
printers_point = printers_points = \
    UnitLength('printers_point', 3.514598e-4*m) # TODO: check
pica = picas = printers_pica = printers_picas = \
    UnitLength('pica', 12*printers_point)

US_survey_foot = US_survey_feet = \
    UnitLength('US_survey_foot', 1200*m/3937)
US_survey_yard = US_survey_yards = \
    UnitLength('US_survey_yard', 3*US_survey_foot)
US_survey_mile = US_survey_miles = \
    UnitLength('US_survey_mile', 5280*US_survey_foot)
US_statute_mile = US_statute_miles = \
    UnitLength('US_statute_mile', US_survey_mile)
rod = rods = pole = poles = perch = perches = \
    UnitLength('rod', 16.5*US_survey_foot) # TODO: check, google uses foot
furlong = furlongs = \
    UnitLength('furlong', 660*US_survey_foot)
fathom = fathoms = \
    UnitLength('fathom', 6*US_survey_foot)
chain = chains = \
    UnitLength('chain', 2.011684e1*m) # TODO: check
big_point = big_points = \
    UnitLength('big_point', inch/72)
barleycorn = barleycorns = \
    UnitLength('barleycorn', inch/3)
arpentlin = \
    UnitLength('arpentlin', 191.835*foot)
