# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


e = elementary_charge = UnitConstant(
    'elementary_charge',
    _cd('elementary charge'),
    symbol='e'
)
elementary_charge_over_h = UnitConstant(
    'elementary_charge_over_h',
    _cd('elementary charge over h'),
    symbol='e/h'
)
Faraday_constant = UnitConstant(
    'Faraday_constant',
    _cd('Faraday constant'),
    symbol='F'
)
#F_star = Faraday_constant_for_conventional_electric_current = UnitConstant(
#    _cd('Faraday constant for conventional electric current') what is a unit of C_90?
r_e = classical_electron_radius = UnitConstant(
    'classical_electron_radius',
    _cd('classical electron radius'),
    symbol='r_e',
    u_symbol='rₑ'
)
m_e = electron_mass = UnitConstant(
    'electron_mass',
    _cd('electron mass'),
    symbol='m_e',
    u_symbol='mₑ'
)
lambda_C = Compton_wavelength = UnitConstant(
    'Compton_wavelength',
    _cd('Compton wavelength'),
    symbol='lambda_C',
    u_symbol='λ_C'
)
Compton_wavelength_over_2_pi = UnitConstant(
    'Compton_wavelength_over_2_pi',
    _cd('Compton wavelength over 2 pi'),
    symbol='lambdabar_C',
    u_symbol='ƛ_C'
)
electron_charge_to_mass_quotient = UnitConstant(
    'electron_charge_to_mass_quotient',
    _cd('electron charge to mass quotient'),
    symbol='(-e/m_e)',
    u_symbol='(-e/mₑ)'
)
g_e = electron_g_factor = UnitConstant(
    'electron_g_factor',
    _cd('electron g factor'),
    symbol='g_e',
    u_symbol='gₑ'
)
gamma_e = electron_gyromagnetic_ratio = UnitConstant(
    'electron_gyromagnetic_ratio',
    _cd('electron gyromagnetic ratio'),
    symbol='gamma_e',
    u_symbol='γₑ'
)
electron_gyromagnetic_ratio_over_2_pi = UnitConstant(
    'electron_gyromagnetic_ratio_over_2_pi',
    _cd('electron gyromagnetic ratio over 2 pi'),
    symbol='gamma_e/(2*pi)',
    u_symbol='γₑ/(2·π)'
)
mu_e = electron_magnetic_moment = UnitConstant(
    'electron_magnetic_moment',
    _cd('electron magnetic moment'),
    symbol='mu_e',
    u_symbol='μₑ'
)
a_e = electron_magnetic_moment_anomaly = UnitConstant(
    'electron_magnetic_moment_anomaly',
    _cd('electron magnetic moment anomaly'),
    symbol='a_e',
    u_symbol='aₑ'
)
eV = electron_volt = UnitConstant(
    'electron_volt',
    _cd('electron volt'),
    symbol='eV'
)
sigma_e = Thomson_cross_section = UnitConstant(
    'Thomson_cross_section',
    _cd('Thomson cross section'),
    symbol='sigma_e',
    u_symbol='σₑ'
)

mu_B = Bohr_magneton = UnitConstant(
    'Bohr_magneton',
    _cd('Bohr magneton'),
    symbol='mu_B',
    u_symbol='μ_B'
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

electron_mass_energy_equivalent = UnitConstant(
    'electron_mass_energy_equivalent',
    _cd('electron mass energy equivalent'),
    symbol='(m_e*c**2)',
    u_symbol='(mₑ·c²)'
)
electron_mass_energy_equivalent_in_MeV = UnitConstant(
    'electron_mass_energy_equivalent_in_MeV',
    _cd('electron mass energy equivalent in MeV')
)
electron_mass_in_u = UnitConstant(
    'electron_mass_in_u',
    _cd('electron mass in u')
)
electron_molar_mass = UnitConstant(
    'electron_molar_mass',
    _cd('electron molar mass'),
    symbol='M_e',
    u_symbol='Mₑ'
)

electron_deuteron_mass_ratio = UnitConstant(
    'electron_deuteron_mass_ratio',
    _cd('electron-deuteron mass ratio'),
    symbol='(m_e/m_d)',
    u_symbol='(mₑ/m_d)'
)
electron_muon_mass_ratio = UnitConstant(
    'electron_muon_mass_ratio',
    _cd('electron-muon mass ratio'),
    symbol='(m_e/m_mu)',
    u_symbol='(mₑ/m_μ)'
)
electron_neutron_mass_ratio = UnitConstant(
    'electron_neutron_mass_ratio',
    _cd('electron-neutron mass ratio'),
    symbol='(m_e/m_n)',
    u_symbol='(mₑ/m_n)'
)
electron_proton_mass_ratio = UnitConstant(
    'electron_proton_mass_ratio',
    _cd('electron-proton mass ratio'),
    symbol='(m_e/m_p)',
    u_symbol='(mₑ/m_p)'
)
electron_tau_mass_ratio = UnitConstant(
    'electron_tau_mass_ratio',
    _cd('electron-tau mass ratio'),
    symbol='(m_e/m_tau)',
    u_symbol='(mₑ/m_τ)'
)
electron_to_alpha_particle_mass_ratio = UnitConstant(
    'electron_to_alpha_particle_mass_ratio',
    _cd('electron to alpha particle mass ratio'),
    symbol='(m_e/m_alpha)',
    u_symbol='(mₑ/m_α)'
)

electron_deuteron_magnetic_moment_ratio = UnitConstant(
    'electron_deuteron_magnetic_moment_ratio',
    _cd('electron-deuteron magnetic moment ratio'),
    symbol='(mu_e/mu_d)',
    u_symbol='(μₑ/μ_d)'
)
electron_magnetic_moment_to_Bohr_magneton_ratio = UnitConstant(
    'electron_magnetic_moment_to_Bohr_magneton_ratio',
    _cd('electron magnetic moment to Bohr magneton ratio'),
    symbol='(mu_e/mu_B)',
    u_symbol='(μₑ/μ_B)'
)
electron_magnetic_moment_to_nuclear_magneton_ratio = UnitConstant(
    'electron_magnetic_moment_to_nuclear_magneton_ratio',
    _cd('electron magnetic moment to nuclear magneton ratio'),
    symbol='(mu_e/mu_N)',
    u_symbol='(μₑ/μ_N)'
)
electron_muon_magnetic_moment_ratio = UnitConstant(
    'electron_muon_magnetic_moment_ratio',
    _cd('electron-muon magnetic moment ratio'),
    symbol='(mu_e/mu_mu)',
    u_symbol='(μₑ/μ_μ)'
)
electron_neutron_magnetic_moment_ratio = UnitConstant(
    'electron_neutron_magnetic_moment_ratio',
    _cd('electron-neutron magnetic moment ratio'),
    symbol='(mu_e/mu_n)',
    u_symbol='(μₑ/μ_n)'
)
electron_proton_magnetic_moment_ratio = UnitConstant(
    'electron_proton_magnetic_moment_ratio',
    _cd('electron-proton magnetic moment ratio'),
    symbol='(mu_e/mu_p)',
    u_symbol='(μₑ/μ_p)'
)
electron_to_shielded_helion_magnetic_moment_ratio = UnitConstant(
    'electron_to_shielded_helion_magnetic_moment_ratio',
    _cd('electron to shielded helion magnetic moment ratio'),
    symbol='(mu_e/muprime_h)',
    u_symbol='(μₑ/μ′_h)'
)
electron_to_shielded_proton_magnetic_moment_ratio = UnitConstant(
    'electron_to_shielded_proton_magnetic_moment_ratio',
    _cd('electron to shielded proton magnetic moment ratio'),
    symbol='(mu_e/muprime_p)',
    u_symbol='(μₑ/μ′_p)'
)


electron_volt_atomic_mass_unit_relationship = UnitConstant(
    'electron_volt_atomic_mass_unit_relationship',
    _cd('electron volt-atomic mass unit relationship')
)
electron_volt_hartree_relationship = UnitConstant(
    'electron_volt_hartree_relationship',
    _cd('electron volt-hartree relationship')
)
electron_volt_hertz_relationship = UnitConstant(
    'electron_volt_hertz_relationship',
    _cd('electron volt-hertz relationship')
)
electron_volt_inverse_meter_relationship = UnitConstant(
    'electron_volt_inverse_meter_relationship',
    _cd('electron volt-inverse meter relationship')
)
electron_volt_joule_relationship = UnitConstant(
    'electron_volt_joule_relationship',
    _cd('electron volt-joule relationship')
)
electron_volt_kelvin_relationship = UnitConstant(
    'electron_volt_kelvin_relationship',
    _cd('electron volt-kelvin relationship')
)
electron_volt_kilogram_relationship = UnitConstant(
    'electron_volt_kilogram_relationship',
    _cd('electron volt-kilogram relationship')
)
hertz_electron_volt_relationship = UnitConstant(
    'hertz_electron_volt_relationship',
    _cd('hertz-electron volt relationship')
)
inverse_meter_electron_volt_relationship = UnitConstant(
    'inverse_meter_electron_volt_relationship',
    _cd('inverse meter-electron volt relationship')
)
joule_electron_volt_relationship = UnitConstant(
    'joule_electron_volt_relationship',
    _cd('joule-electron volt relationship')
)
kelvin_electron_volt_relationship = UnitConstant(
    'kelvin_electron_volt_relationship',
    _cd('kelvin-electron volt relationship')
)
kilogram_electron_volt_relationship = UnitConstant(
    'kilogram_electron_volt_relationship',
    _cd('kilogram-electron volt relationship')
)

del UnitConstant, _cd
