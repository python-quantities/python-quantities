# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


m_n = neutron_mass = UnitConstant(
    'neutron_mass',
    _cd('neutron mass'),
    symbol='m_n'
)
lambda_C_n = neutron_Compton_wavelength = UnitConstant(
    'neutron_Compton_wavelength',
    _cd('neutron Compton wavelength'),
    symbol='lambda_C_n',
    u_symbol='λ_C_n'
)
neutron_Compton_wavelength_over_2_pi = UnitConstant(
    'neutron_Compton_wavelength_over_2_pi',
    _cd('neutron Compton wavelength over 2 pi'),
    symbol='lambdabar_C_n',
    u_symbol='ƛ_C_n'
)
g_n = neutron_g_factor = UnitConstant(
    'neutron_g_factor',
    _cd('neutron g factor'),
    symbol='g_n'
)
gamma_n = neutron_gyromagnetic_ratio = UnitConstant(
    'neutron_gyromagnetic_ratio',
    _cd('neutron gyromagnetic ratio'),
    symbol='gamma_n',
    u_symbol='γ_n'
)
neutron_gyromagnetic_ratio_over_2_pi = UnitConstant(
    'neutron_gyromagnetic_ratio_over_2_pi',
    _cd('neutron gyromagnetic ratio over 2 pi'),
    symbol='(gamma_n/(2*pi))',
    u_symbol='(γ_n/(2·π))'
)
mu_n = neutron_magnetic_moment = UnitConstant(
    'neutron_magnetic_moment',
    _cd('neutron magnetic moment'),
    symbol='mu_n',
    u_symbol='μ_n'
)
neutron_mass_energy_equivalent = UnitConstant(
    'neutron_mass_energy_equivalent',
    _cd('neutron mass energy equivalent'),
    symbol='(m_n*c**2)',
    u_symbol='(m_n·c²)'
)
neutron_mass_energy_equivalent_in_MeV = UnitConstant(
    'neutron_mass_energy_equivalent_in_MeV',
    _cd('neutron mass energy equivalent in MeV')
)
neutron_mass_in_u = UnitConstant(
    'neutron_mass_in_u',
    _cd('neutron mass in u')
)
neutron_molar_mass = UnitConstant(
    'neutron_molar_mass',
    _cd('neutron molar mass'),
    symbol='M_n'
)

neutron_electron_mass_ratio = UnitConstant(
    'neutron_electron_mass_ratio',
    _cd('neutron-electron mass ratio'),
    symbol='(m_n/m_e)',
    u_symbol='(m_n/mₑ)'
)
neutron_muon_mass_ratio = UnitConstant(
    'neutron_muon_mass_ratio',
    _cd('neutron-muon mass ratio'),
    symbol='(m_n/m_mu)',
    u_symbol='(m_n/m_μ)'
)
neutron_proton_mass_ratio = UnitConstant(
    'neutron_proton_mass_ratio',
    _cd('neutron-proton mass ratio'),
    symbol='(m_n/m_p)',
    u_symbol='(m_n/m_p)'
)
neutron_tau_mass_ratio = UnitConstant(
    'neutron_tau_mass_ratio',
    _cd('neutron-tau mass ratio'),
    symbol='(m_n/m_tau)',
    u_symbol='(m_n/m_τ)'
)


neutron_electron_magnetic_moment_ratio = UnitConstant(
    'neutron_electron_magnetic_moment_ratio',
    _cd('neutron-electron magnetic moment ratio'),
    symbol='(mu_n/mu_e)',
    u_symbol='(μ_n/μₑ)'
)
neutron_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'neutron_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('neutron magnetic moment to Bohr magneton ratio'),
    symbol='(mu_n/mu_B)',
    u_symbol='(μ_n/μ_B)'
)
neutron_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'neutron_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('neutron magnetic moment to nuclear magneton ratio'),
    symbol='(mu_n/mu_N)',
    u_symbol='(μ_n/μ_N)'
)
neutron_proton_magnetic_moment_ratio = UnitConstant(
    'neutron_proton_magnetic_moment_ratio',
    _cd('neutron-proton magnetic moment ratio'),
    symbol='(mu_n/mu_p)',
    u_symbol='(μ_n/μ_p)'
)
neutron_to_shielded_proton_magnetic_moment_ratio = UnitConstant(
    'neutron_to_shielded_proton_magnetic_moment_ratio',
    _cd('neutron to shielded proton magnetic moment ratio'),
    symbol='(mu_n/muprime_p)',
    u_symbol='(μ_n/μ′_p)'
)

del UnitConstant, _cd
