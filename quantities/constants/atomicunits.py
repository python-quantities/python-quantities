# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


amu = atomic_mass_constant = UnitConstant(
    'atomic_mass_constant',
    _cd('atomic mass constant'),
    symbol='m_u',
    u_symbol='mᵤ'
)
atomic_unit_of_1st_hyperpolarizability = UnitConstant(
    'atomic_unit_of_1st_hyperpolarizability',
    _cd('atomic unit of 1st hyperpolarizability'),
    symbol='(e**3*a_0**3/E_h**2)',
    u_symbol='(e³·a₀³/E_h²)'
)
atomic_unit_of_2nd_hyperpolarizability = UnitConstant(
    'atomic_unit_of_2nd_hyperpolarizability',
    _cd('atomic unit of 2nd hyperpolarizability'),
    symbol='(e**4*a_0**4/E_h**3)',
    u_symbol='(e⁴·a₀⁴/E_h³)'
)
hbar = atomic_unit_of_action = UnitConstant(
    'atomic_unit_of_action',
    _cd('atomic unit of action'),
    symbol='hbar',
    u_symbol='ħ'
)
atomic_unit_of_charge = UnitConstant(
    'atomic_unit_of_charge',
    _cd('atomic unit of charge'),
    symbol='e'
)
atomic_unit_of_charge_density = UnitConstant(
    'atomic_unit_of_charge_density',
    _cd('atomic unit of charge density'),
    symbol='(e/a_0**3)',
    u_symbol='(e/a₀³)'
)
atomic_unit_of_current = UnitConstant(
    'atomic_unit_of_current',
    _cd('atomic unit of current'),
    symbol='(e*E_h/hbar)',
    u_symbol='(e·E_h/ħ)'
)
atomic_unit_of_electric_dipole_moment = UnitConstant(
    'atomic_unit_of_electric_dipole_moment',
    _cd('atomic unit of electric dipole moment'),
    symbol='(e*a_0)',
    u_symbol='(e·a₀)'
)
atomic_unit_of_electric_field = UnitConstant(
    'atomic_unit_of_electric_field',
    _cd('atomic unit of electric field'),
    symbol='(E_h/(e*a_0))',
    u_symbol='(E_h/(e·a₀))'
)
atomic_unit_of_electric_field_gradient = UnitConstant(
    'atomic_unit_of_electric_field_gradient',
    _cd('atomic unit of electric field gradient'),
    symbol='(E_h/(e*a_0**2))',
    u_symbol='(E_h/(e·a₀²))'
)
atomic_unit_of_electric_polarizability = UnitConstant(
    'atomic_unit_of_electric_polarizability',
    _cd('atomic unit of electric polarizability'),
    symbol='(e**2*a_0**2/E_h)',
    u_symbol='(e²·a₀²/E_h)'
)
atomic_unit_of_electric_potential = UnitConstant(
    'atomic_unit_of_electric_potential',
    _cd('atomic unit of electric potential'),
    symbol='(E_h/e)'
)
atomic_unit_of_electric_quadrupole_moment = UnitConstant(
    'atomic_unit_of_electric_quadrupole_moment',
    _cd('atomic unit of electric quadrupole moment'),
    symbol='(e*a_0**2)',
    u_symbol='(e·a₀²)'
)
atomic_unit_of_energy = UnitConstant(
    'atomic_unit_of_energy',
    _cd('atomic unit of energy'),
)
atomic_unit_of_force = UnitConstant(
    'atomic_unit_of_force',
    _cd('atomic unit of force'),
    symbol='(E_h/a_0)',
    u_symbol='(E_h/a₀)'
)
a_0 = atomic_unit_of_length = UnitConstant(
    'atomic_unit_of_length',
    _cd('atomic unit of length'),
    symbol='a_0',
    u_symbol='a₀'
)
Bohr_radius = UnitConstant(
    'Bohr_radius',
    _cd('Bohr radius'),
    symbol='a_0',
    u_symbol='a₀'
)
atomic_unit_of_magnetic_dipole_moment = UnitConstant(
    'atomic_unit_of_magnetic_dipole_moment',
    _cd('atomic unit of magnetic dipole moment'),
    symbol='(hbar*e/m_e)',
    u_symbol='(ħ·e/mₑ)'
)
atomic_unit_of_magnetic_flux_density = UnitConstant(
    'atomic_unit_of_magnetic_flux_density',
    _cd('atomic unit of magnetic flux density'),
    symbol='(hbar*e/a_0**2)',
    u_symbol='(ħ·e/a₀²)'
)
atomic_unit_of_magnetizability = UnitConstant(
    'atomic_unit_of_magnetizability',
    _cd('atomic unit of magnetizability'),
    symbol='(e**2a_0**2/m_e)',
    u_symbol='(e²·a₀²/mₑ)'
)
m_e = atomic_unit_of_mass = UnitConstant(
    'atomic_unit_of_mass',
    _cd('atomic unit of mass'),
    symbol='m_e',
    u_symbol='mₑ'
)
atomic_unit_of_momentum = UnitConstant(
    'atomic_unit_of_momentum',
    _cd('atomic unit of momentum'),
    symbol='(hbar/a_0)',
    u_symbol='(ħ/a₀)'
)
atomic_unit_of_permittivity = UnitConstant(
    'atomic_unit_of_permittivity',
    _cd('atomic unit of permittivity'),
    symbol='(e**2/(a_0*E_h))',
    u_symbol='(e²/(a₀·E_h))'
)
atomic_unit_of_time = UnitConstant(
    'atomic_unit_of_time',
    _cd('atomic unit of time'),
    symbol='(hbar/E_h)',
    u_symbol='(ħ/E_h)'
)
atomic_unit_of_velocity = UnitConstant(
    'atomic_unit_of_velocity',
    _cd('atomic unit of velocity'),
    symbol='(a_0*E_h/hbar)',
    u_symbol='(a₀·E_h/ħ)'
)
E_h = Hartree_energy = UnitConstant(
    'Hartree_energy',
    _cd('Hartree energy'),
    symbol='E_h'
)
u = unified_atomic_mass_unit = UnitConstant(
    'unified_atomic_mass_unit',
    _cd('unified atomic mass unit'),
    symbol='u'
)
molar_mass_of_carbon_12 = UnitConstant(
    'molar_mass_of_carbon_12',
    _cd('molar mass of carbon-12'),
    symbol='M_C12',
    u_symbol='M_¹²C'
)

atomic_mass_constant_energy_equivalent = UnitConstant(
    'atomic_mass_constant_energy_equivalent',
    _cd('atomic mass constant energy equivalent'),
    symbol='(m_u*c**2)',
    u_symbol='(mᵤ·c²)'
)
atomic_mass_constant_energy_equivalent_in_MeV = UnitConstant(
    'atomic_mass_constant_energy_equivalent_in_MeV',
    _cd('atomic mass constant energy equivalent in MeV')
)
Hartree_energy_in_eV = UnitConstant(
    'Hartree_energy_in_eV',
    _cd('Hartree energy in eV')
)

atomic_mass_unit_electron_volt_relationship = UnitConstant(
    'atomic_mass_unit_electron_volt_relationship',
    _cd('atomic mass unit-electron volt relationship'),
)
atomic_mass_unit_hartree_relationship = UnitConstant(
    'atomic_mass_unit_hartree_relationship',
    _cd('atomic mass unit-hartree relationship')
)
atomic_mass_unit_hertz_relationship = UnitConstant(
    'atomic_mass_unit_hertz_relationship',
    _cd('atomic mass unit-hertz relationship')
)
atomic_mass_unit_inverse_meter_relationship = UnitConstant(
    'atomic_mass_unit_inverse_meter_relationship',
    _cd('atomic mass unit-inverse meter relationship')
)
atomic_mass_unit_joule_relationship = UnitConstant(
    'atomic_mass_unit_joule_relationship',
    _cd('atomic mass unit-joule relationship'),
    symbol='(u*c**2)',
    u_symbol='(u·c²)'
)
atomic_mass_unit_kelvin_relationship = UnitConstant(
    'atomic_mass_unit_kelvin_relationship',
    _cd('atomic mass unit-kelvin relationship')
)
atomic_mass_unit_kilogram_relationship = UnitConstant(
    'atomic_mass_unit_kilogram_relationship',
    _cd('atomic mass unit-kilogram relationship')
)

hartree_atomic_mass_unit_relationship = UnitConstant(
    'hartree_atomic_mass_unit_relationship',
    _cd('hartree-atomic mass unit relationship')
)
hartree_electron_volt_relationship = UnitConstant(
    'hartree_electron_volt_relationship',
    _cd('hartree-electron volt relationship')
)
hartree_hertz_relationship = UnitConstant(
    'hartree_hertz_relationship',
    _cd('hartree-hertz relationship')
)
hartree_inverse_meter_relationship = UnitConstant(
    'hartree_inverse_meter_relationship',
    _cd('hartree-inverse meter relationship')
)
hartree_joule_relationship = UnitConstant(
    'hartree_joule_relationship',
    _cd('hartree-joule relationship')
)
hartree_kelvin_relationship = UnitConstant(
    'hartree_kelvin_relationship',
    _cd('hartree-kelvin relationship')
)
hartree_kilogram_relationship = UnitConstant(
    'hartree_kilogram_relationship',
    _cd('hartree-kilogram relationship')
)
hertz_atomic_mass_unit_relationship = UnitConstant(
    'hertz_atomic_mass_unit_relationship',
    _cd('hertz-atomic mass unit relationship')
)
hertz_hartree_relationship = UnitConstant(
    'hertz_hartree_relationship',
    _cd('hertz-hartree relationship')
)
inverse_meter_atomic_mass_unit_relationship = UnitConstant(
    'inverse_meter_atomic_mass_unit_relationship',
    _cd('inverse meter-atomic mass unit relationship')
)
inverse_meter_hartree_relationship = UnitConstant(
    'inverse_meter_hartree_relationship',
    _cd('inverse meter-hartree relationship')
)
joule_atomic_mass_unit_relationship = UnitConstant(
    'joule_atomic_mass_unit_relationship',
    _cd('joule-atomic mass unit relationship')
)
joule_hartree_relationship = UnitConstant(
    'joule_hartree_relationship',
    _cd('joule-hartree relationship')
)
kelvin_atomic_mass_unit_relationship = UnitConstant(
    'kelvin_atomic_mass_unit_relationship',
    _cd('kelvin-atomic mass unit relationship')
)
kelvin_hartree_relationship = UnitConstant(
    'kelvin_hartree_relationship',
    _cd('kelvin-hartree relationship')
)
kilogram_atomic_mass_unit_relationship = UnitConstant(
    'kilogram_atomic_mass_unit_relationship',
    _cd('kilogram-atomic mass unit relationship')
)
kilogram_hartree_relationship = UnitConstant(
    'kilogram_hartree_relationship',
    _cd('kilogram-hartree relationship')
)

del UnitConstant, _cd
