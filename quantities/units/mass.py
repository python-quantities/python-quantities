"""
"""

from quantities.units.unitquantities import compound, mass
from quantities.units.length import m

kg = kilogram = kilograms = mass('kg')
gram = grams = mass('gram')
mg = milligram = milligrams = mass('mg')
ounce = ounces = avoirdupois_ounce = mass('avoirdupois_ounce')
lb = lbs = pound = pounds = avoirdupois_pound = mass('pound')

assay_ton = mass('assay_ton')
carat = mass('carat')
gr = grain = grains = mass('gr')
long_hundredweight = mass('long_hundredweight')
t = metric_ton = tonne = mass('t')
pennyweight = mass('pennyweight')
short_hundredweight = mass('short_hundredweight')
slug = mass('slug')
apothecary_ounce = troy_ounce = mass('troy_ounce')
apothecary_pound = troy_pound = mass('troy_pound')
u = amu = atomic_mass_unit = mass('amu')
scruple = mass('scruple')
apdram = mass('apdram')
apounce = mass('apounce')
appound = mass('appound')
bag = mass('bag')
ton = short_ton = mass('short_ton')
long_ton = mass('long_ton')

############################################################
##                 Mass per unit length                   ##
############################################################

denier = deniers = 1.111111e-7 * kg/m
tex = texs = 1e-6 * kg/m

denier_ = deniers_ = compound('denier')
tex_ = texs_ = compound('tex')

del compound, m
