# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


Z_0 = impedence_of_free_space = characteristic_impedance_of_vacuum = UnitConstant(
    'characteristic_impedance_of_vacuum',
    _cd('characteristic impedance of vacuum'),
    symbol='Z_0',
    u_symbol='Z₀'
)
vacuum_permittivity = epsilon_0 = electric_constant = UnitConstant(
    'electric_constant',
    _cd('electric constant'),
    symbol='epsilon_0',
    u_symbol='ε₀'
)
mu_0 = magnetic_constant = UnitConstant(
    'magnetic_constant',
    _cd('magnetic constant'),
    symbol='mu_0',
    u_symbol='μ₀'
)


del UnitConstant, _cd
