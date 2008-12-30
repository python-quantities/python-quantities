"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.mass import gram, kg
from quantities.units.length import cm, m
from quantities.units.time import s
from quantities.units.acceleration import g


N = newton = newtons = UnitQuantity('N', kg*m/s**2)
dyne = dynes = UnitQuantity('dyne', gram*cm/s**2)
pond = ponds = UnitQuantity('pond', 9.806650e-3*N)
kgf = force_kilogram = kilogram_force = UnitQuantity('kgf', kg*g)
ozf = force_ounce = ounce_force = UnitQuantity('ozf', 2.780139e-1*N)
lbf = force_pound = pound_force = UnitQuantity('lbf', 4.4482216152605*N)
poundal = poundals = UnitQuantity('poundal', 1.382550e-1*N)
gf = gram_force = force_gram = UnitQuantity('gf', gram*g)
force_ton = ton_force = UnitQuantity('force_ton', 2000*force_pound)
kip = UnitQuantity('kip', 1000*lbf)

del UnitQuantity, gram, kg, cm, m, s, g
