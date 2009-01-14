"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.time import s
from quantities.units.mass import kg
from quantities.units.energy import J
from quantities.units.electromagnetism import coulomb


Bq = becquerel = UnitQuantity(
    'becquerel',
    1/s,
    symbol='Bq',
    aliases=['becquerels']
)
Ci = curie = UnitQuantity(
    'curie',
    3.7e10*becquerel,
    symbol='Ci',
    aliases=['curies']
)
rd = rutherford = UnitQuantity(
    'rutherford',
    1e6*Bq,
    symbol='Rd',
    aliases=['rutherfords'],
    note='this unit is obsolete, in favor of 1e6 Bq'
)
Gy = gray = Sv = sievert = UnitQuantity(
    'gray',
    J/kg,
    symbol='Gy',
    aliases=['grays', 'Sv', 'sievert', 'sieverts']
)
rem = UnitQuantity(
    'rem',
    1e-2*sievert,
    aliases='rems'
)
rad = rads = UnitQuantity(
    'rad',
    1e-2*gray,
    aliases=['rads']
)
R = roentgen = UnitQuantity(
    'roentgen',
    2.58e-4*coulomb/kg,
    symbol='R',
    aliases=['roentgens']
)

del UnitQuantity, s, kg, J, coulomb
