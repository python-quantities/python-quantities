"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .length import cm, m, foot, inch
from .area import acre

l = L = liter = litre = UnitQuantity(
    'liter',
    1e-3*m**3,
    symbol='L',
    aliases=['l', 'liters', 'litre', 'litres']
)
mL = milliliter = millilitre = UnitQuantity(
    'milliliter',
    liter/1000,
    symbol='mL',
    aliases=['ml', 'milliliters', 'millilitre', 'millilitres']
)
kL = kiloliter = kilolitre = UnitQuantity(
    'kiloliter',
    liter*1000,
    symbol='kL',
    aliases=['kl', 'kiloliters', 'kilolitre', 'kilolitres']
)
ML = megaliter = megalitre = UnitQuantity(
    'megaliter',
    kiloliter*1000,
    symbol='ML',
    aliases=['Ml', 'megaliters', 'megalitre', 'megalitres']
)
GL = gigaliter = gigalitre = UnitQuantity(
    'gigaliter',
    megaliter*1000,
    symbol='GL',
    aliases=['Gl', 'gigaliters', 'gigalitre', 'gigalitres']
)
cc = cubic_centimeter = milliliter = UnitQuantity(
    'cubic_centimeter',
    cm**3,
    symbol='cc',
    aliases=['cubic_centimeters']
)

stere = UnitQuantity(
    'stere',
    m**3,
    aliases=['steres']
)
gross_register_ton = register_ton = UnitQuantity(
    'gross_register_ton',
    100*foot**3,
    symbol='GRT',
    aliases=['gross_register_tons', 'register_ton', 'register_tons']
)

acre_foot = UnitQuantity(
    'acre_foot',
    acre*foot,
    aliases=['acre_feet']
)
board_foot = UnitQuantity(
    'board_foot',
    foot**2*inch,
    symbol='FBM',
    aliases=['board_feet'])
bu = bushel = US_bushel = UnitQuantity(
    'US_bushel',
    2150.42*inch**3,
    symbol='bu'
)
US_dry_gallon = UnitQuantity(
    'US_dry_gallon',
    bushel/8,
)
gallon = liquid_gallon = US_liquid_gallon = UnitQuantity(
    'US_liquid_gallon',
    231*inch**3,
    aliases=[
        'gallon', 'gallons', 'liquid_gallon', 'liquid_gallons',
        'US_liquid_gallons'
    ],
)
dry_quart = US_dry_quart = UnitQuantity(
    'US_dry_quart',
    US_dry_gallon/4,
    aliases=['dry_quart', 'dry_quarts', 'US_dry_quarts']
)
dry_pint = US_dry_pint = UnitQuantity(
    'US_dry_pint',
    US_dry_quart/2,
    aliases=['dry_pint', 'dry_pints', 'US_dry_pints']
)
quart = liquid_quart = US_liquid_quart = UnitQuantity(
    'US_liquid_quart',
    US_liquid_gallon/4,
    symbol='quart',
    aliases=['quarts', 'liquid_quart', 'liquid_quarts', 'US_liquid_quarts']
)
pt = pint = liquid_pint = US_liquid_pint = UnitQuantity(
    'US_liquid_pint',
    US_liquid_quart/2,
    symbol='pt',
    aliases=[
        'pint', 'pints', 'liquid_pint', 'liquid_pints', 'US_liquid_pints'
    ],
)
cup = US_liquid_cup = UnitQuantity(
    'cup',
    US_liquid_pint/2,
    aliases=['cups', 'US_liquid_cup', 'US_liquid_cups']
)
gill = US_liquid_gill = UnitQuantity(
    'US_liquid_gill',
    US_liquid_cup/2,
    symbol='gill',
    aliases=['gills', 'US_liquid_gills']
)
floz = fluid_ounce = US_fluid_ounce = US_liquid_ounce = UnitQuantity(
    'US_fluid_ounce',
    US_liquid_gill/4,
    symbol='fl_oz',
    aliases=[
        'fluid_ounce', 'fluid_ounces', 'US_fluid_ounces', 'US_liquid_ounce',
        'US_liquid_ounces'
    ]
)

Imperial_bushel = UnitQuantity(
    'Imperial_bushel',
    36.36872*liter,
    doc='exact'
)
UK_liquid_gallon = Canadian_liquid_gallon = UnitQuantity(
    'UK_liquid_gallon',
    4.54609*liter,
    aliases=[
        'UK_liquid_gallons', 'Canadian_liquid_gallon',
        'Canadian_liquid_gallons'
    ],
    doc='exact'
)
UK_liquid_quart = UnitQuantity(
    'UK_liquid_quart',
    UK_liquid_gallon/4,
    aliases=['UK_liquid_quarts']
)
UK_liquid_pint = UnitQuantity(
    'UK_liquid_pint',
    UK_liquid_quart/2,
    aliases=['UK_liquid_pints']
)
UK_liquid_cup = UnitQuantity(
    'UK_liquid_cup',
    UK_liquid_pint/2,
    aliases=['UK_liquid_cups']
)
UK_liquid_gill = UnitQuantity(
    'UK_liquid_gill',
    UK_liquid_cup/2,
    aliases=['UK_liquid_gills']
)
UK_fluid_ounce = UK_liquid_ounce = UnitQuantity(
    'UK_fluid_ounce',
    UK_liquid_gill/5, # not a mistake
    aliases=['UK_fluid_ounces', 'UK_liquid_ounce', 'UK_liquid_ounces']
)

bbl = barrel = UnitQuantity(
    'barrel',
    42*US_liquid_gallon,
    symbol='bbl'
)
tbsp = Tbsp = Tblsp = tblsp = tbs = Tbl = tablespoon = UnitQuantity(
    'tablespoon',
    US_fluid_ounce/2,
    symbol='tbsp',
    aliases=['Tbsp', 'Tblsp', 'Tbl', 'tblsp', 'tbs', 'tablespoons']
)
tsp = teaspoon = UnitQuantity(
    'teaspoon',
    tablespoon/3,
    symbol='tsp',
    aliases=['teaspoons']
)

pk = peck = UnitQuantity(
    'peck',
    bushel/4,
    symbol='pk',
    aliases=['pecks']
)

fldr = fluid_dram = fluidram = UnitQuantity(
    'fluid_dram',
    floz/8,
    symbol='fldr',
    aliases=['fluid_drams', 'fluidram', 'fluidrams']
)

firkin = UnitQuantity(
    'firkin',
    barrel/4
)

del UnitQuantity, cm, m
