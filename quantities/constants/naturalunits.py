# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


natural_unit_of_action = UnitConstant(
    'natural_unit_of_action',
    _cd('natural unit of action'),
    symbol='hbar',
    u_symbol='ħ'
)
natural_unit_of_energy = UnitConstant(
    'natural_unit_of_energy',
    _cd('natural unit of energy'),
    symbol='(m_e*c**2)',
    u_symbol='(mₑ·c²)'
)
natural_unit_of_length = UnitConstant(
    'natural_unit_of_length',
    _cd('natural unit of length'),
    symbol='lambdabar_C',
    u_symbol='ƛ_C'
)
natural_unit_of_mass = UnitConstant(
    'natural_unit_of_mass',
    _cd('natural unit of mass'),
    symbol='m_e',
    u_symbol='mₑ'
)
natural_unit_of_momentum = UnitConstant(
    'natural_unit_of_momentum',
    _cd('natural unit of momentum'),
    symbol='(m_e*c)',
    u_symbol='(mₑ·c)'
)
natural_unit_of_time = UnitConstant(
    'natural_unit_of_time',
    _cd('natural unit of time'),
    symbol='(hbar/(m_e*c**2))',
    u_symbol='(ħ/(mₑ·c²))'
)
natural_unit_of_velocity = UnitConstant(
    'natural_unit_of_velocity',
    _cd('natural unit of velocity'),
    symbol='c'
)

natural_unit_of_action_in_eV_s = UnitConstant(
    'natural_unit_of_action_in_eV_s',
    _cd('natural unit of action in eV s')
)
natural_unit_of_energy_in_MeV = UnitConstant(
    'natural_unit_of_energy_in_MeV',
    _cd('natural unit of energy in MeV')
)
natural_unit_of_momentum_in_MeV_per_c = UnitConstant(
    'natural_unit_of_momentum_in_MeV_per_c',
    _cd('natural unit of momentum in MeV/c')
)

del UnitConstant, _cd
