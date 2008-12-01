"""
"""

from quantities.units.unitquantities import compound
from quantities.units.mass import gram, kg
from quantities.units.length import cm, m
from quantities.units.time import s
from quantities.units.acceleration import g

N = newton = newtons = kg*m/s**2
dyne = dynes = gram*cm/s**2
pond = ponds = 9.806650e-3 * N
kgf = force_kilogram = kilogram_force = 9.806650 * N
ozf = force_ounce = ounce_force = 2.780139e-1 * N
lbf = force_pound = pound_force = 4.4482216152605 * N
poundal = poundals = 1.382550e-1 * N
gf = gram_force = force_gram = gram * g
force_ton = ton_force = 2000 * force_pound
kip = 1000 * lbf

N_ = newton_ = newtons_ = compound('N')
dyne_ = dynes_ = compound('dyne')
pond_ = ponds_ = compound('pond')
kgf_ = force_kilogram_ = kilogram_force_ = compound('kgf')
ozf_ = force_ounce_ = ounce_force_ = compound('ozf')
lbf_ = force_pound_ = pound_force_ = compound('lbf')
poundal_ = poundals_ = compound('poundal')
gf_ = gram_force_ = force_gram_ = compound('gf')
force_ton_ = ton_force_ = compound('force_ton')
kip_ = compound('kip')

del compound, gram, kg, cm, m, s, g
