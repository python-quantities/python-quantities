"""
"""

from quantities.units.unitquantity import UnitQuantity, UnitMass
from quantities.units.length import m

kg = kilogram = kilograms = UnitMass('kg')
gram = grams = UnitMass('gram')
mg = milligram = milligrams = UnitMass('mg')
ounce = ounces = avoirdupois_ounce = UnitMass('avoirdupois_ounce')
lb = lbs = pound = pounds = avoirdupois_pound = UnitMass('pound')

assay_ton = UnitMass('assay_ton')
carat = UnitMass('carat')
gr = grain = grains = UnitMass('gr')
long_hundredweight = UnitMass('long_hundredweight')
t = metric_ton = tonne = UnitMass('t')
pennyweight = UnitMass('pennyweight')
short_hundredweight = UnitMass('short_hundredweight')
slug = UnitMass('slug')
apothecary_ounce = troy_ounce = UnitMass('troy_ounce')
apothecary_pound = troy_pound = UnitMass('troy_pound')
u = amu = atomic_mass_unit = UnitMass('amu')
scruple = UnitMass('scruple')
apdram = UnitMass('apdram')
apounce = UnitMass('apounce')
appound = UnitMass('appound')
bag = UnitMass('bag')
ton = short_ton = UnitMass('short_ton')
long_ton = UnitMass('long_ton')

############################################################
##                 Mass per unit length                   ##
############################################################

denier = deniers = UnitQuantity('denier', 1.111111e-7*kg/m)
tex = texs = UnitQuantity('tex', 1e-6*kg/m)

del UnitQuantity, m
