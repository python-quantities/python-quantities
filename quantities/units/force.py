"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .mass import gram, kg, ounce, lb
from .length import cm, m, ft
from .time import s
from .acceleration import g_0


N = newton = UnitQuantity(
    'newton',
    kg*m/s**2,
    symbol='N',
    aliases=['newtons']
)
dyne = UnitQuantity(
    'dyne',
    gram*cm/s**2,
    symbol='dyn',
    aliases=['dynes']
)
pond = UnitQuantity(
    'pond',
    g_0*kg,
    symbol='p',
    aliases=['ponds']
)
kgf = force_kilogram = kilogram_force = UnitQuantity(
    'kilogram_force',
    kg*g_0,
    symbol='kgf',
    aliases=['force_kilogram']
)
ozf = force_ounce = ounce_force = UnitQuantity(
    'ounce_force',
    ounce*g_0,
    symbol='ozf',
    aliases=['force_ounce']
)
lbf = force_pound = pound_force = UnitQuantity(
    'pound_force',
    lb*g_0,
    symbol='lbf',
    aliases=['force_pound']
)
poundal = UnitQuantity(
    'poundal',
    lb*ft/s**2,
    symbol='pdl',
    aliases=['poundals']
)
gf = gram_force = force_gram = UnitQuantity(
    'gram_force',
    gram*g_0,
    symbol='gf',
    aliases=['force_gram']
)
force_ton = ton_force = UnitQuantity(
    'ton_force',
    2000*force_pound,
    aliases=['force_ton'])
kip = UnitQuantity(
    'kip', 1000*lbf
)

del UnitQuantity, gram, kg, cm, m, s, g_0
