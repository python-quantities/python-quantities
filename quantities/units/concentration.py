"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .substance import mol
from .volume import L

M = molar = UnitQuantity(
    'molar',
    mol / L,
    symbol='M',
    aliases=['molar', 'Molar']
)

mM = millimolar = UnitQuantity(
    'millimolar',
    molar / 1000,
    symbol='mM',
    aliases=['millimolar']
)

uM = micromolar = UnitQuantity(
    'micromolar',
    mM / 1000,
    symbol='uM',
    aliases=['mircomolar']
)