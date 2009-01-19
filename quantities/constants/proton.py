# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


m_p = proton_mass = UnitConstant(
    'proton_mass',
    _cd('proton mass'),
    symbol='m_p'
)
lambda_C_p = proton_Compton_wavelength = UnitConstant(
    'proton_Compton_wavelength',
    _cd('proton Compton wavelength'),
    symbol='lambda_C_p',
    u_symbol='λ_C_p'
)
proton_Compton_wavelength_over_2_pi = UnitConstant(
    'proton_Compton_wavelength_over_2_pi',
    _cd('proton Compton wavelength over 2 pi'),
    symbol='lambdabar_C_p',
    u_symbol='ƛ_C_p'
)
R_p = proton_rms_charge_radius = UnitConstant(
    'proton_rms_charge_radius',
    _cd('proton rms charge radius'),
    symbol='R_p'
)
proton_charge_to_mass_quotient = UnitConstant(
    'proton_charge_to_mass_quotient',
    _cd('proton charge to mass quotient'),
    symbol='(e/m_p)'
)
g_p = proton_g_factor = UnitConstant(
    'proton_g_factor',
    _cd('proton g factor'),
    symbol='g_p'
)
gamma_p = proton_gyromagnetic_ratio = UnitConstant(
    'proton_gyromagnetic_ratio',
    _cd('proton gyromagnetic ratio'),
    symbol='gamma_p',
    u_symbol='γ_p'
)
proton_gyromagnetic_ratio_over_2_pi = UnitConstant(
    'proton_gyromagnetic_ratio_over_2_pi',
    _cd('proton gyromagnetic ratio over 2 pi'),
    symbol='(gamma_p/(2*pi))',
    u_symbol='(γ_p/(2·π))'
)
mu_p = proton_magnetic_moment = UnitConstant(
    'proton_magnetic_moment',
    _cd('proton magnetic moment'),
    symbol='mu_p',
    u_symbol='μ_p'
)
sigma_prime_p = proton_magnetic_shielding_correction = UnitConstant(
    'proton_magnetic_shielding_correction',
    _cd('proton magnetic shielding correction'),
    symbol='sigmaprime_p',
    u_symbol='σ′_p'
)
gamma_prime_p = shielded_proton_gyromagnetic_ratio = UnitConstant(
    'shielded_proton_gyromagnetic_ratio',
    _cd('shielded proton gyromagnetic ratio'),
    symbol='gammaprime_p',
    u_symbol='γ′_p'
)
shielded_proton_gyromagnetic_ratio_over_2_pi = UnitConstant(
    'shielded_proton_gyromagnetic_ratio_over_2_pi',
    _cd('shielded proton gyromagnetic ratio over 2 pi'),
    symbol='(gammaprime_p/(2*pi))',
    u_symbol='(γ′_p/(2·π))'
)
mu_prime_p = shielded_proton_magnetic_moment = UnitConstant(
    'shielded_proton_magnetic_moment',
    _cd('shielded proton magnetic moment'),
    symbol='muprime_p',
    u_symbol='μ′_p'
)

mu_N = nuclear_magneton = UnitConstant(
    'nuclear_magneton',
    _cd('nuclear magneton'),
    symbol='mu_N',
    u_symbol='μ_N'
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

proton_mass_energy_equivalent = UnitConstant(
    'proton_mass_energy_equivalent',
    _cd('proton mass energy equivalent'),
    symbol='(m_p*c**2)',
    u_symbol='(m_p·c²)'
)
proton_mass_energy_equivalent_in_MeV = UnitConstant(
    'proton_mass_energy_equivalent_in_MeV',
    _cd('proton mass energy equivalent in MeV')
)
proton_mass_in_u = UnitConstant(
    'proton_mass_in_u',
    _cd('proton mass in u')
)
proton_molar_mass = UnitConstant(
    'proton_molar_mass',
    _cd('proton molar mass'),
    symbol='M_p'
)

proton_electron_mass_ratio = UnitConstant(
    'proton_electron_mass_ratio',
    _cd('proton-electron mass ratio'),
    symbol='(m_p/m_e)',
    u_symbol='(m_p/mₑ)'
)
proton_muon_mass_ratio = UnitConstant(
    'proton_muon_mass_ratio',
    _cd('proton-muon mass ratio'),
    symbol='(m_p/m_mu)',
    u_symbol='(m_p/m_μ)'
)
proton_neutron_mass_ratio = UnitConstant(
    'proton_neutron_mass_ratio',
    _cd('proton-neutron mass ratio'),
    symbol='(m_p/m_n)',
)
proton_tau_mass_ratio = UnitConstant(
    'proton_tau_mass_ratio',
    _cd('proton-tau mass ratio'),
    symbol='(m_p/m_tau)',
    u_symbol='(m_p/m_τ)'
)

proton_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'proton_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('proton magnetic moment to Bohr magneton ratio'),
    symbol='(mu_p/mu_B)',
    u_symbol='(μ_p/μ_B)'
)
proton_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'proton_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('proton magnetic moment to nuclear magneton ratio'),
    symbol='(mu_p/mu_N)',
    u_symbol='(μ_p/μ_N)'
)
proton_neutron_magnetic_moment_ratio = UnitConstant(
    'proton_neutron_magnetic_moment_ratio',
    _cd('proton-neutron magnetic moment ratio'),
    symbol='(mu_p/mu_n)',
    u_symbol='(μ_p/μ_n)'
)
shielded_proton_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'shielded_proton_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('shielded proton magnetic moment to Bohr magneton ratio'),
    symbol='(muprime_p/mu_B)',
    u_symbol='(μ′_p/μ_B)'
)
shielded_proton_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'shielded_proton_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('shielded proton magnetic moment to nuclear magneton ratio'),
    symbol='(muprime_p/mu_N)',
    u_symbol='(μ′_p/μ_N)'
)

del UnitConstant, _cd
