# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

#import math as _math
from .utils import _cd
from quantities.unitquantity import UnitConstant


Bohr_magneton_in_eV_per_T = UnitConstant(
    'Bohr_magneton_in_eV_per_T',
    _cd('Bohr magneton in eV/T')
)
Bohr_magneton_in_Hz_per_T = UnitConstant(
    'Bohr_magneton_in_Hz_per_T',
    _cd('Bohr magneton in Hz/T')
)
Bohr_magneton_in_inverse_meters_per_tesla = UnitConstant(
    'Bohr_magneton_in_inverse_meters_per_tesla',
    _cd('Bohr magneton in inverse meters per tesla')
)
Bohr_magneton_in_K_per_T = UnitConstant(
    'Bohr_magneton_in_K_per_T',
    _cd('Bohr magneton in K/T')
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
Hartree_energy_in_eV = UnitConstant(
    'Hartree_energy_in_eV',
    _cd('Hartree energy in eV')
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
nuclear_magneton_in_eV_per_T = UnitConstant(
    'nuclear_magneton_in_eV_per_T',
    _cd('nuclear magneton in eV/T')
)
nuclear_magneton_in_inverse_meters_per_tesla = UnitConstant(
    'nuclear_magneton_in_inverse_meters_per_tesla',
    _cd('nuclear magneton in inverse meters per tesla')
)
nuclear_magneton_in_K_per_T = UnitConstant(
    'nuclear_magneton_in_K_per_T',
    _cd('nuclear magneton in K/T')
)
nuclear_magneton_in_MHz_per_T = UnitConstant(
    'nuclear_magneton_in_MHz_per_T',
    _cd('nuclear magneton in MHz/T')
)
Planck_constant_in_eV_s = UnitConstant(
    'Planck_constant_in_eV_s',
    _cd('Planck constant in eV s')
)
Rydberg_constant_times_c_in_Hz = UnitConstant(
    'Rydberg_constant_times_c_in_Hz',
    _cd('Rydberg constant times c in Hz')
)
Rydberg_constant_times_hc_in_eV = UnitConstant(
    'Rydberg_constant_times_hc_in_eV',
    _cd('Rydberg constant times hc in eV')
)
Rydberg_constant_times_hc_in_J = UnitConstant(
    'Rydberg_constant_times_hc_in_J',
    _cd('Rydberg constant times hc in J'),
    symbol='(R_infinity*h*c)',
    u_symbol='(R_∞·h·c)'
)

del UnitConstant, _cd
