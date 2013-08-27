# -*- coding: utf-8 -*-
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
    aliases=['Molar']
)

mM = millimolar = UnitQuantity(
    'millimolar',
    molar / 1000,
    symbol='mM'
)

uM = micromolar = UnitQuantity(
    'micromolar',
    mM / 1000,
    symbol='uM',
    u_symbol='µM'
)
