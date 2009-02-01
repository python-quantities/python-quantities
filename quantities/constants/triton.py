# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


m_t = triton_mass = UnitConstant(
    'triton_mass',
    _cd('triton mass'),
    symbol='m_t'
)
g_t = triton_g_factor = UnitConstant(
    'triton_g_factor',
    _cd('triton g factor'),
    symbol='g_t'
)
mu_t = triton_magnetic_moment = UnitConstant(
    'triton_magnetic_moment',
    _cd('triton magnetic moment'),
    symbol='mu_t',
    u_symbol='μ_t'
)
triton_mass_energy_equivalent = UnitConstant(
    'triton_mass_energy_equivalent',
    _cd('triton mass energy equivalent'),
    symbol='(m_t*c**2)',
    u_symbol='(m_t·c²)'
)
triton_mass_energy_equivalent_in_MeV = UnitConstant(
    'triton_mass_energy_equivalent_in_MeV',
    _cd('triton mass energy equivalent in MeV')
)
triton_mass_in_u = UnitConstant(
    'triton_mass_in_u',
    _cd('triton mass in u')
)
triton_molar_mass = UnitConstant(
    'triton_molar_mass',
    _cd('triton molar mass'),
    symbol='M_t'
)

triton_electron_mass_ratio = UnitConstant(
    'triton_electron_mass_ratio',
    _cd('triton-electron mass ratio'),
    symbol='(m_t/m_e)',
    u_symbol='(m_t/mₑ)'
)
triton_proton_mass_ratio = UnitConstant(
    'triton_proton_mass_ratio',
    _cd('triton-proton mass ratio'),
    symbol='(m_t/m_p)',
    u_symbol='(m_t/m_p)'
)

triton_electron_magnetic_moment_ratio = UnitConstant(
    'triton_electron_magnetic_moment_ratio',
    _cd('triton-electron magnetic moment ratio'),
    symbol='(mu_t/mu_e)',
    u_symbol='(μ_t/μₑ)'
)
triton_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'triton_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('triton magnetic moment to Bohr magneton ratio'),
    symbol='(mu_t/mu_B)',
    u_symbol='(μ_t/μ_B)'
)
triton_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'triton_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('triton magnetic moment to nuclear magneton ratio'),
    symbol='(mu_t/mu_N)',
    u_symbol='(μ_t/μ_N)'
)
triton_neutron_magnetic_moment_ratio = UnitConstant(
    'triton_neutron_magnetic_moment_ratio',
    _cd('triton-neutron magnetic moment ratio'),
    symbol='(mu_t/mu_n)',
    u_symbol='(μ_t/μ_n)'
)
triton_proton_magnetic_moment_ratio = UnitConstant(
    'triton_proton_magnetic_moment_ratio',
    _cd('triton-proton magnetic moment ratio'),
    symbol='(mu_t/mu_p)',
    u_symbol='(μ_t/μ_p)'
)

del UnitConstant, _cd
