"""
"""

from quantities.units.unitquantities import compound
from quantities.units.length import cm, m

acre_foot = acre_feet = 1.233489e3 * m**3
board_foot = board_feet = 2.359737e-3 * m**3
bu = bushel = bushels = 3.523907e-2 * m**3
UK_liquid_gallon = UK_liquid_gallons = 4.546090e-3 * m**3
Canadian_liquid_gallon = Canadian_liquid_gallons = UK_liquid_gallon
US_dry_gallon = US_dry_gallons = 4.404884e-3 * m**3
gallon = gallons = liquid_gallon = liquid_gallons = US_liquid_gallon = US_liquid_gallons = 3.785412e-3 * m**3
cc = cm**3
l = L = liter = liters = litre = litres = 1e-3 * m**3
stere = steres = m**3
register_ton = register_tons = 2.831685 * m**3
dry_quart = dry_quarts = US_dry_quart = US_dry_quarts = US_dry_gallon/4
dry_pint = dry_pints = US_dry_pint = US_dry_pints = US_dry_gallon/8
quart = liquid_quart = US_liquid_quart = US_liquid_quarts = US_liquid_gallon/4
pt = pint = pints = liquid_pint = liquid_pints = US_liquid_pint = US_liquid_pints = US_liquid_gallon/8
cup = cups = US_liquid_cup = US_liquid_cups = US_liquid_gallon/16
gill = gills = US_liquid_gill = US_liquid_gills = US_liquid_gallon/32
oz = floz = fluid_ounce = fluid_ounces = US_fluid_ounce = US_fluid_ounces = US_liquid_ounce = US_liquid_ounces = US_liquid_gallon/128

UK_liquid_quart = UK_liquid_quarts = UK_liquid_gallon/4
UK_liquid_pint = UK_liquid_pints = UK_liquid_gallon/8
UK_liquid_cup = UK_liquid_cups = UK_liquid_gallon/16
UK_liquid_gill = UK_liquid_gills = UK_liquid_gallon/32
UK_fluid_ounce = UK_fluid_ounces = UK_liquid_ounce = UK_liquid_ounces = UK_liquid_gallon/160

bbl = barrel = barrels = 42 * US_liquid_gallon # petroleum industry definition
tbsp = Tbsp = Tblsp = tblsp = Tbl = tablespoon = tablespoons = US_fluid_ounce/2
tsp = teaspoon = teaspoons = tablespoon/3
pk = peck = pecks = bushel/4

fldr = floz/8
dr = dram = floz/16

firkin = barrel/4 # exact but barrel is vague

acre_foot_ = acre_feet_ = compound('acre_feet')
board_foot_ = board_feet_ = compound('board_feet')
bu_ = bushel_ = bushels_ = compound('bu')
UK_liquid_gallon_ = UK_liquid_gallons_ = compound('UK_liquid_gallon')
Canadian_liquid_gallon_ = Canadian_liquid_gallons_ = UK_liquid_gallon
US_dry_gallon_ = US_dry_gallons_ = compound('US_dry_gallon')
gallon_ = gallons_ = liquid_gallon_ = liquid_gallons_ = US_liquid_gallon_ = US_liquid_gallons_ = compound('gallon')
cc_ = compound('cc')
l_ = L_ = liter_ = liters_ = litre_ = litres_ = compound('l')
stere_ = steres_ = compound('stere')
register_ton_ = register_tons_ = compound('register_ton')
dry_quart_ = dry_quarts_ = US_dry_quart_ = US_dry_quarts_ = compound('dry_quart')
dry_pint_ = dry_pints_ = US_dry_pint_ = US_dry_pints_ = compound('dry_pint')
quart_ = liquid_quart_ = US_liquid_quart_ = US_liquid_quarts_ = compound('quart')
pt_ = pint_ = pints_ = liquid_pint_ = liquid_pints_ = US_liquid_pint_ = US_liquid_pints = compound('pt')
cup_ = cups_ = US_liquid_cup_ = US_liquid_cups_ = compound('cup')
gill_ = gills_ = US_liquid_gill_ = US_liquid_gills_ = compound('gill')
oz_ = floz_ = fluid_ounce_ = fluid_ounces_ = US_fluid_ounce_ = US_fluid_ounces_ = US_liquid_ounce_ = US_liquid_ounces_ = compound('oz')

UK_liquid_quart_ = UK_liquid_quarts_ = compound('UK_liquid_quart')
UK_liquid_pint_ = UK_liquid_pints_ = compound('UK_liquid_pint')
UK_liquid_cup_ = UK_liquid_cups_ = compound('UK_liquid_cup')
UK_liquid_gill_ = UK_liquid_gills_ = compound('UK_liquid_gill')
UK_fluid_ounce_ = UK_fluid_ounces_ = UK_liquid_ounce_ = UK_liquid_ounces_ = compound('UK_fluid_ounce')

bbl_ = barrel_ = barrels_ = compound('bbl')
tbsp_ = Tbsp_ = Tblsp_ = tblsp_ = Tbl_ = tablespoon_ = tablespoons_ = compound('tbsp')
tsp_ = teaspoon_ = teaspoons_ = compound('tsp')
pk_ = peck_ = pecks_ = compound('pk')

fldr_ = compound('fldr')
dr_ = dram_ = compound('dr')

firkin_ = compound('firkin')

del compound, cm, m
