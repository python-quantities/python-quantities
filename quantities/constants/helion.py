# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant



m_h = helion_mass = UnitConstant(
    'helion_mass',
    _cd('helion mass'),
    symbol='m_h'
)
gamma_prime_h = shielded_helion_gyromagnetic_ratio = UnitConstant(
    'shielded_helion_gyromagnetic_ratio',
    _cd('shielded helion gyromagnetic ratio'),
    symbol='gammaprime_h',
    u_symbol='γ′_h'
)
shielded_helion_gyromagnetic_ratio_over_2_pi = UnitConstant(
    'shielded_helion_gyromagnetic_ratio_over_2_pi',
    _cd('shielded helion gyromagnetic ratio over 2 pi'),
    symbol='(gammaprime_h/(2*pi))',
    u_symbol='(γ′_h/(2·π))'
)
mu_prime_h = shielded_helion_magnetic_moment = UnitConstant(
    'shielded_helion_magnetic_moment',
    _cd('shielded helion magnetic moment'),
    symbol='muprime_h',
    u_symbol='μ′_h'
)
helion_mass_energy_equivalent = UnitConstant(
    'helion_mass_energy_equivalent',
    _cd('helion mass energy equivalent'),
    symbol='(m_h*c**2)',
    u_symbol='(m_h·c²)'
)
helion_mass_energy_equivalent_in_MeV = UnitConstant(
    'helion_mass_energy_equivalent_in_MeV',
    _cd('helion mass energy equivalent in MeV')
)
helion_mass_in_u = UnitConstant(
    'helion_mass_in_u',
    _cd('helion mass in u')
)
helion_molar_mass = UnitConstant(
    'helion_molar_mass',
    _cd('helion molar mass'),
    symbol='M_h'
)
helion_electron_mass_ratio = UnitConstant(
    'helion_electron_mass_ratio',
    _cd('helion-electron mass ratio'),
    symbol='(m_h/m_e)',
    u_symbol='(m_h/mₑ)'
)
helion_proton_mass_ratio = UnitConstant(
    'helion_proton_mass_ratio',
    _cd('helion-proton mass ratio'),
    symbol='(m_h/m_p)'
)

shielded_helion_to_proton_magnetic_moment_ratio = UnitConstant(
    'shielded_helion_to_proton_magnetic_moment_ratio',
    _cd('shielded helion to proton magnetic moment ratio'),
    symbol='(muprime_h/mu_p)',
    u_symbol='(μ′_h/μ_p)'
)
shielded_helion_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'shielded_helion_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('shielded helion magnetic moment to Bohr magneton ratio'),
    symbol='(muprime_h/mu_B)',
    u_symbol='(μ′_h/μ_B)'
)
shielded_helion_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'shielded_helion_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('shielded helion magnetic moment to nuclear magneton ratio'),
    symbol='(muprime_h/mu_N)',
    u_symbol='(μ′_h/μ_N)'
)
shielded_helion_to_shielded_proton_magnetic_moment_ratio = UnitConstant(
    'shielded_helion_to_shielded_proton_magnetic_moment_ratio',
    _cd('shielded helion to shielded proton magnetic moment ratio'),
    symbol='(muprime_h/muprime_p)',
    u_symbol='(μ′_h/μ′_p)'
)

del UnitConstant, _cd
