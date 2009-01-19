# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


m_mu = muon_mass = UnitConstant(
    'muon_mass',
    _cd('muon mass'),
    symbol='m_mu',
    u_symbol='m_μ'
)
lambda_C_mu = muon_Compton_wavelength = UnitConstant(
    'muon_Compton_wavelength',
    _cd('muon Compton wavelength'),
    symbol='lambdabar_C_mu',
    u_symbol='λ_C_μ'
)
muon_Compton_wavelength_over_2_pi = UnitConstant(
    'muon_Compton_wavelength_over_2_pi',
    _cd('muon Compton wavelength over 2 pi'),
    symbol='lambdabar_Cmu',
    u_symbol='ƛ_C_μ'
)
g_mu = muon_g_factor = UnitConstant(
    'muon_g_factor',
    _cd('muon g factor'),
    symbol='g_mu',
    u_symbol='g_μ'
)
mu_mu = muon_magnetic_moment = UnitConstant(
    'muon_magnetic_moment',
    _cd('muon magnetic moment'),
    symbol='mu_mu',
    u_symbol='μ_μ'
)
a_mu = muon_magnetic_moment_anomaly = UnitConstant(
    'muon_magnetic_moment_anomaly',
    _cd('muon magnetic moment anomaly'),
    symbol='a_mu',
    u_symbol='a_μ'
)
muon_mass_energy_equivalent = UnitConstant(
    'muon_mass_energy_equivalent',
    _cd('muon mass energy equivalent'),
    symbol='(m_mu*c**2)',
    u_symbol='(m_μ·c²)'
)
muon_mass_energy_equivalent_in_MeV = UnitConstant(
    'muon_mass_energy_equivalent_in_MeV',
    _cd('muon mass energy equivalent in MeV')
)
muon_mass_in_u = UnitConstant(
    'muon_mass_in_u',
    _cd('muon mass in u')
)
muon_molar_mass = UnitConstant(
    'muon_molar_mass',
    _cd('muon molar mass'),
    symbol='M_mu',
    u_symbol='M_μ'
)

muon_electron_mass_ratio = UnitConstant(
    'muon_electron_mass_ratio',
    _cd('muon-electron mass ratio'),
    symbol='(m_mu/m_e)',
    u_symbol='(m_μ/mₑ)'
)
muon_neutron_mass_ratio = UnitConstant(
    'muon_neutron_mass_ratio',
    _cd('muon-neutron mass ratio'),
    symbol='(m_mu/m_n)',
    u_symbol='(m_μ/m_n)'
)
muon_proton_mass_ratio = UnitConstant(
    'muon_proton_mass_ratio',
    _cd('muon-proton mass ratio'),
    symbol='(m_mu/m_p)',
    u_symbol='(m_μ/m_p)'
)
muon_tau_mass_ratio = UnitConstant(
    'muon_tau_mass_ratio',
    _cd('muon-tau mass ratio'),
    symbol='(m_mu/m_tau)',
    u_symbol='(m_μ/m_τ)'
)

muon_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'muon_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('muon magnetic moment to Bohr magneton ratio'),
    symbol='(mu_mu/mu_B)',
    u_symbol='(μ_μ/μ_B)'
)
muon_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'muon_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('muon magnetic moment to nuclear magneton ratio'),
    symbol='(mu_mu/mu_N)',
    u_symbol='(μ_μ/μ_N)'
)
muon_proton_magnetic_moment_ratio = UnitConstant(
    'muon_proton_magnetic_moment_ratio',
    _cd('muon-proton magnetic moment ratio'),
    symbol='(mu_mu/mu_p)',
    u_symbol='(μ_μ/μ_p)'
)

del UnitConstant, _cd
