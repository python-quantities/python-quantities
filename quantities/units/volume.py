"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.length import cm, m


acre_foot = acre_feet = \
    UnitQuantity('acre_feet', 1.233489e3*m**3)
board_foot = board_feet = \
    UnitQuantity('board_feet', 2.359737e-3*m**3)
bu = bushel = bushels = \
    UnitQuantity('bu', 3.523907e-2*m**3)
UK_liquid_gallon = UK_liquid_gallons = Canadian_liquid_gallon = \
    Canadian_liquid_gallons = UK_liquid_gallon = \
    UnitQuantity('UK_liquid_gallon', 4.546090e-3*m**3)
US_dry_gallon = US_dry_gallons = \
    UnitQuantity('US_dry_gallon', 4.404884e-3*m**3)
gallon = gallons = liquid_gallon = liquid_gallons = US_liquid_gallon = \
    US_liquid_gallons = \
    UnitQuantity('gallon', 3.785412e-3*m**3)
cc = UnitQuantity('cc', cm**3)
l = L = liter = liters = litre = litres = \
    UnitQuantity('l', 1e-3*m**3)
stere = steres = UnitQuantity('stere', m**3)
register_ton = register_tons = \
    UnitQuantity('register_ton', 2.831685*m**3)
dry_quart = dry_quarts = US_dry_quart = US_dry_quarts = \
    UnitQuantity('dry_quart', US_dry_gallon/4)
dry_pint = dry_pints = US_dry_pint = US_dry_pints = \
    UnitQuantity('dry_pint', US_dry_gallon/8)
quart = liquid_quart = US_liquid_quart = US_liquid_quarts = \
    UnitQuantity('quart', US_liquid_gallon/4)
pt = pint = pints = liquid_pint = liquid_pints = US_liquid_pint = \
    US_liquid_pints = \
    UnitQuantity('pt', US_liquid_gallon/8)
cup = cups = US_liquid_cup = US_liquid_cups = \
    UnitQuantity('cup', US_liquid_gallon/16)
gill = gills = US_liquid_gill = US_liquid_gills = \
    UnitQuantity('gill', US_liquid_gallon/32)
oz = floz = fluid_ounce = fluid_ounces = US_fluid_ounce = US_fluid_ounces = \
    US_liquid_ounce = US_liquid_ounces = \
    UnitQuantity('oz', US_liquid_gallon/128)

UK_liquid_quart = UK_liquid_quarts = \
    UnitQuantity('UK_liquid_quart', UK_liquid_gallon/4)
UK_liquid_pint = UK_liquid_pints = \
    UnitQuantity('UK_liquid_pint', UK_liquid_gallon/8)
UK_liquid_cup = UK_liquid_cups = \
    UnitQuantity('UK_liquid_cup', UK_liquid_gallon/16)
UK_liquid_gill = UK_liquid_gills = \
    UnitQuantity('UK_liquid_gill', UK_liquid_gallon/32)
UK_fluid_ounce = UK_fluid_ounces = UK_liquid_ounce = UK_liquid_ounces = \
    UnitQuantity('UK_fluid_ounce', UK_liquid_gallon/160)

bbl = barrel = barrels = \
    UnitQuantity('bbl', 42*US_liquid_gallon)
tbsp = Tbsp = Tblsp = tblsp = Tbl = tablespoon = tablespoons = \
    UnitQuantity('tbsp', US_fluid_ounce/2)
tsp = teaspoon = teaspoons = \
    UnitQuantity('tsp', tablespoon/3)
pk = peck = pecks = \
    UnitQuantity('pk', bushel/4)

fldr = \
    UnitQuantity('fldr', floz/8)
dr = dram = \
    UnitQuantity('dr', floz/16)

firkin = \
    UnitQuantity('firkin', barrel/4)

del UnitQuantity, cm, m
