# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


m_alpha = alpha_particle_mass = UnitConstant(
    'alpha_particle_mass',
    _cd('alpha particle mass'),
    symbol='m_alpha',
    u_symbol='m_α'
)
alpha_particle_mass_energy_equivalent = UnitConstant(
    'alpha_particle_mass_energy_equivalent',
    _cd('alpha particle mass energy equivalent'),
    symbol='(m_alpha*c**2)',
    u_symbol='(m_α·c²)'
)
alpha_particle_mass_energy_equivalent_in_MeV = UnitConstant(
    'alpha_particle_mass_energy_equivalent_in_MeV',
    _cd('alpha particle mass energy equivalent in MeV'),
)
alpha_particle_mass_in_u = UnitConstant(
    'alpha_particle_mass_in_u',
    _cd('alpha particle mass in u')
)
alpha_particle_molar_mass = UnitConstant(
    'alpha_particle_molar_mass',
    _cd('alpha particle molar mass'),
    symbol='M_alpha',
    u_symbol='M_α'
)
alpha_particle_electron_mass_ratio = UnitConstant(
    'alpha_particle_electron_mass_ratio',
    _cd('alpha particle-electron mass ratio'),
    symbol='(m_alpha/m_e)',
    u_symbol='(m_α/mₑ)'
)
alpha_particle_proton_mass_ratio = UnitConstant(
    'alpha_particle_proton_mass_ratio',
    _cd('alpha particle-proton mass ratio'),
    symbol='(m_alpha/m_p)',
    u_symbol='(m_α/m_p)'
)

del UnitConstant, _cd
