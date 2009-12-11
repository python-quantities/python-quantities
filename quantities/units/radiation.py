"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .time import s
from .mass import kg
from .energy import J
from .electromagnetism import coulomb


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
    doc='this unit is obsolete, in favor of 1e6 Bq'
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
    aliases=['rems']
)
rads = UnitQuantity(
    'rads',
    1e-2*gray,
    doc='''
    rad is commonly used symbol for radian. 
    rads unit of radiation is deprecated.
    '''
)
R = roentgen = UnitQuantity(
    'roentgen',
    2.58e-4*coulomb/kg,
    symbol='R',
    aliases=['roentgens']
)

del UnitQuantity, s, kg, J, coulomb
