# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


m_tau = tau_mass = UnitConstant(
    'tau_mass',
    _cd('tau mass'),
    symbol='m_tau',
    u_symbol='m_τ'
)
lambda_C_tau = tau_Compton_wavelength = UnitConstant(
    'tau_Compton_wavelength',
    _cd('tau Compton wavelength'),
    symbol='lambda_C_tau',
    u_symbol='λ_C_τ'
)
tau_Compton_wavelength_over_2_pi = UnitConstant(
    'tau_Compton_wavelength_over_2_pi',
    _cd('tau Compton wavelength over 2 pi'),
    symbol='lambda_C_tau',
    u_symbol='ƛ_C_τ'
)
tau_mass_energy_equivalent = UnitConstant(
    'tau_mass_energy_equivalent',
    _cd('tau mass energy equivalent'),
    symbol='(m_tau*c**2)',
    u_symbol='(m_τ·c²)'
)
tau_mass_energy_equivalent_in_MeV = UnitConstant(
    'tau_mass_energy_equivalent_in_MeV',
    _cd('tau mass energy equivalent in MeV')
)
tau_mass_in_u = UnitConstant(
    'tau_mass_in_u',
    _cd('tau mass in u')
)
tau_molar_mass = UnitConstant(
    'tau_molar_mass',
    _cd('tau molar mass'),
    symbol='M_tau',
    u_symbol='M_τ'
)

tau_electron_mass_ratio = UnitConstant(
    'tau_electron_mass_ratio',
    _cd('tau-electron mass ratio'),
    symbol='(m_tau/m_e)',
    u_symbol='(m_τ/mₑ)'
)
tau_muon_mass_ratio = UnitConstant(
    'tau_muon_mass_ratio',
    _cd('tau-muon mass ratio'),
    symbol='(m_tau/m_mu)',
    u_symbol='(m_τ/m_μ)'
)
tau_neutron_mass_ratio = UnitConstant(
    'tau_neutron_mass_ratio',
    _cd('tau-neutron mass ratio'),
    symbol='(m_tau/m_n)',
    u_symbol='(m_τ/m_n)'
)
tau_proton_mass_ratio = UnitConstant(
    'tau_proton_mass_ratio',
    _cd('tau-proton mass ratio'),
    symbol='(m_tau/m_p)',
    u_symbol='(m_τ/m_p)'
)

del UnitConstant, _cd
