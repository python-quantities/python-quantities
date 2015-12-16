# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitSubstance

mol = mole = UnitSubstance(
    'mole',
    symbol='mol'
)
mmol = UnitSubstance(
    'millimole',
    mol/1000,
    symbol='mmol'
)
umol = UnitSubstance(
    'micromole',
    mmol/1000,
    symbol='umol',
    u_symbol='µmol'
)
