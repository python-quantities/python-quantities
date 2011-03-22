# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


R_d = deuteron_rms_charge_radius = UnitConstant(
    'deuteron_rms_charge_radius',
    _cd('deuteron rms charge radius'),
    symbol='R_d'
)
m_d = deuteron_mass = UnitConstant(
    'deuteron_mass',
    _cd('deuteron mass'),
    symbol='m_d'
)
g_d = deuteron_g_factor = UnitConstant(
    'deuteron_g_factor',
    _cd('deuteron g factor'),
    symbol='g_d'
)
mu_d = deuteron_magnetic_moment = UnitConstant(
    'deuteron_magnetic_moment',
    _cd('deuteron magnetic moment'),
    symbol='mu_d',
    u_symbol='μ_d'
)
deuteron_mass_energy_equivalent = UnitConstant(
    'deuteron_mass_energy_equivalent',
    _cd('deuteron mass energy equivalent'),
    symbol='(m_d*c**2)',
    u_symbol='(m_d·c²)'
)
deuteron_mass_energy_equivalent_in_MeV = UnitConstant(
    'deuteron_mass_energy_equivalent_in_MeV',
    _cd('deuteron mass energy equivalent in MeV')
)
deuteron_mass_in_u = UnitConstant(
    'deuteron_mass_in_u',
    _cd('deuteron mass in u')
)
deuteron_molar_mass = UnitConstant(
    'deuteron_molar_mass',
    _cd('deuteron molar mass'),
    symbol='M_d'
)
deuteron_electron_mass_ratio = UnitConstant(
    'deuteron_electron_mass_ratio',
    _cd('deuteron-electron mass ratio'),
    symbol='(m_d/m_e)',
    u_symbol='(m_d/mₑ)'
)
deuteron_proton_mass_ratio = UnitConstant(
    'deuteron_proton_mass_ratio',
    _cd('deuteron-proton mass ratio'),
    symbol='(m_d/m_n)'
)
deuteron_electron_magnetic_moment_ratio = UnitConstant(
    'deuteron_electron_magnetic_moment_ratio',
    _cd('deuteron-electron magnetic moment ratio'),
    symbol='(mu_d/mu_e)',
    u_symbol='(μ_d/μₑ)'
)
deuteron_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'deuteron_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('deuteron magnetic moment to Bohr magneton ratio'),
    symbol='(mu_d/mu_B)',
    u_symbol='(μ_d/μ_B)'
)
deuteron_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'deuteron_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('deuteron magnetic moment to nuclear magneton ratio'),
    symbol='(mu_d/mu_N)',
    u_symbol='(μ_d/μ_N)'
)
deuteron_neutron_magnetic_moment_ratio = UnitConstant(
    'deuteron_neutron_magnetic_moment_ratio',
    _cd('deuteron-neutron magnetic moment ratio'),
    symbol='(mu_d/mu_n)',
    u_symbol='(μ_d/μ_n)'
)
deuteron_proton_magnetic_moment_ratio = UnitConstant(
    'deuteron_proton_magnetic_moment_ratio',
    _cd('deuteron-proton magnetic moment ratio'),
    symbol='(mu_d/mu_p)',
    u_symbol='(μ_d/μ_p)'
)

del UnitConstant, _cd
