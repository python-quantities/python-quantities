# -*- coding: utf-8 -*-
"""
"""

from quantities.units.unitquantity import UnitQuantity, UnitMass
from quantities.units.length import m

kg = kilogram =  UnitMass(
    'kilograms',
    symbol='kg',
    aliases=['kilograms']
)
g = gram = UnitMass(
    'gram',
    kg/1000,
    symbol='g',
    aliases=['grams']
)
mg = milligram = UnitMass(
    'milligram',
    gram/1000,
    symbol='mg',
    aliases=['milligrams']
)
ounce = avoirdupois_ounce = UnitMass(
    'ounce',
    28.349523125*g,
    symbol='oz',
    aliases=['ounces','avoirdupois_ounce', 'avoirdupois_ounces']
) # exact
lb = pound = avoirdupois_pound = UnitMass(
    'pound',
    0.45359237*kg,
    symbol='lb',
    aliases=['pounds', 'avoirdupois_pound', 'avoirdupois_pounds']
) # exact

carat = UnitMass(
    'carat',
    200*mg,
    aliases=['carats']
)
gr = grain = UnitMass(
    'grain',
    64.79891*mg,
    symbol='gr',
    aliases=['grains']
)
long_hundredweight = UnitMass(
    'long_hundredweight',
    112*lb,
    aliases=['long_hundredweights']
)
short_hundredweight = UnitMass(
    'short_hundredweight',
    100*lb,
    aliases=['short_hundredweights']
) # cwt is used for both short and long hundredweight, so we wont use it
t = metric_ton = tonne = UnitMass(
    'tonne',
    1000*kg,
    symbol='t',
    aliases=['tonnes']
)
dwt = pennyweight = UnitMass(
    'pennyweight',
    24*gr,
    symbol='dwt',
    aliases=['pennyweights']
)
slug = slugs = UnitMass(
    'slug',
    14.59390*kg,
    aliases=['slugs']
)
toz = troy_ounce = apounce = apothecary_ounce = UnitMass(
    'troy_ounce',
    480*gr,
    symbol='℥',
    aliases=[
        'apounce', 'apounces', 'apothecary_ounce', 'apothecary_ounces',
        'troy_ounces'
    ]
)
troy_pound = appound = apothecary_pound = UnitMass(
    'troy_pound',
    12*toz,
    symbol='℔',
    aliases=[
        'troy_pounds', 'appound', 'appounds', 'apothecary_pound',
        'apothecary_pounds'
    ]
)
u = amu = atomic_mass_unit = UnitMass(
    'atomic_mass_unit',
    1.660538782e-27*kg,
    symbol='u',
    aliases=['amu']
) # TODO: needs uncertainty: 0.000000083e-27*kg
scruple = UnitMass(
    'scruple',
    20*gr,
    symbol='℈',
    aliases=['scruples']
)
dram = drachm = apdram = UnitMass(
    'drachm',
    60*gr,
    symbol='ʒ',
    aliases=['dram', 'drams', 'drachms', 'apdrams']
)

bag = UnitMass(
    'bag',
    94*lb,
    aliases=['bags']
)

ton = short_ton = UnitMass(
    'short_ton',
    2000*lb,
    aliases=['short_tons']
)
long_ton = UnitMass(
    'long_ton', 2240*lb,
    aliases=['long_tons']
) # both long and short tons are referred to as "ton" so we wont use it

############################################################
##                 Mass per unit length                   ##
############################################################

denier = UnitQuantity(
    'denier',
    g/(9000*m),
    aliases=['deniers']
)
tex = UnitQuantity(
    'tex',
    g/(1000*m),
    aliases=['texs']
)
dtex = UnitQuantity(
    'dtex',
    g/(10000*m),
    aliases=['dtexs']
)

del UnitQuantity, m
