# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


N_A = L = Avogadro_constant = UnitConstant(
    'Avogadro_constant',
    _cd('Avogadro constant'),
    symbol='N_A'
)
n_0 = Loschmidt_constant = UnitConstant(
    'Loschmidt_constant',
    _cd('Loschmidt constant (273.15 K, 101.325 kPa)'),
    symbol='n_0',
    u_symbol='n₀'
)
R = molar_gas_constant = UnitConstant(
    'molar_gas_constant',
    _cd('molar gas constant'),
    symbol='R'
)

k = Boltzmann_constant = UnitConstant(
    'Boltzmann_constant',
    _cd('Boltzmann constant'),
    symbol='k'
)
Boltzmann_constant_in_eV_per_K = UnitConstant(
    'Boltzmann_constant_in_eV_per_K',
    _cd('Boltzmann constant in eV/K')
)
Boltzmann_constant_in_Hz_per_K = UnitConstant(
    'Boltzmann_constant_in_Hz_per_K',
    _cd('Boltzmann constant in Hz/K')
)
Boltzmann_constant_in_inverse_meters_per_kelvin = UnitConstant(
    'Boltzmann_constant_in_inverse_meters_per_kelvin',
    _cd('Boltzmann constant in inverse meters per kelvin')
)

M_u = molar_mass_constant = UnitConstant(
    'molar_mass_constant',
    _cd('molar mass constant'),
    symbol='M_u',
    u_symbol='Mᵤ'
)
molar_volume_of_ideal_gas_ST_100kPa = UnitConstant(
    'molar_volume_of_ideal_gas_ST_100kPa',
    _cd('molar volume of ideal gas (273.15 K, 100 kPa)')
)
molar_volume_of_ideal_gas_STP = UnitConstant(
    'molar_volume_of_ideal_gas_STP',
    _cd('molar volume of ideal gas (273.15 K, 101.325 kPa)')
)
molar_volume_of_silicon = UnitConstant(
    'molar_volume_of_silicon',
    _cd('molar volume of silicon')
)

del UnitConstant, _cd
