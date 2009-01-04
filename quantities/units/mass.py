"""
"""

from quantities.units.unitquantity import UnitQuantity, UnitMass
from quantities.units.length import m

kg = kilogram = kilograms = \
    UnitMass('kg')
gram = grams = \
    UnitMass('gram', kg/1000)
mg = milligram = milligrams = \
    UnitMass('mg', gram/1000)
ounce = ounces = avoirdupois_ounce = \
    UnitMass('avoirdupois_ounce', 2.834952e-2*kg)
lb = lbs = pound = pounds = avoirdupois_pound = \
    UnitMass('pound', 4.5359237e-1*kg)

assay_ton = \
    UnitMass('assay_ton', 2.916667e-2*kg)
carat = \
    UnitMass('carat', 2e-4*kg)
gr = grain = grains = \
    UnitMass('gr', 6.479891e-5*kg)
long_hundredweight = long_hundredweights = \
    UnitMass('long_hundredweight', 50.80235*kg)
t = metric_ton = tonne = \
    UnitMass('t', 1000*kg)
pennyweight = pennyweights = \
    UnitMass('pennyweight', 1.555174e-3*kg)
short_hundredweight = short_hundredweights = \
    UnitMass('short_hundredweight', 45.35924*kg)
slug = slugs = \
    UnitMass('slug', 14.59390*kg)
apothecary_ounce = troy_ounce = \
    UnitMass('troy_ounce', 3.110348e-2*kg)
apothecary_pound = troy_pound = \
    UnitMass('troy_pound', 0.3732417*kg)
amu = atomic_mass_unit = u = \
    UnitMass('amu', 1.66053886e-27*kg)
u = \
    UnitMass('u', 1.66053886e-27*kg)
scruple = \
    UnitMass('scruple', 20*gr)
apdram = \
    UnitMass('apdram', 60*gr)
apounce = \
    UnitMass('apounce', 480*gr)
appound = \
    UnitMass('appound', 5760*gr)
bag = \
    UnitMass('bag', 94*lb)
ton = short_ton = \
    UnitMass('short_ton', 2000*lb)
long_ton = \
    UnitMass('long_ton', 2240*lb)

############################################################
##                 Mass per unit length                   ##
############################################################

denier = deniers = \
    UnitQuantity('denier', 1.111111e-7*kg/m)
tex = texs = \
    UnitQuantity('tex', 1e-6*kg/m)

del UnitQuantity, m
