# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


molar_Planck_constant = UnitConstant(
    'molar_Planck_constant',
    _cd('molar Planck constant'),
    symbol='(N_A*h)',
    u_symbol='(N_A·h)'
)
molar_Planck_constant_times_c = UnitConstant(
    'molar_Planck_constant_times_c',
    _cd('molar Planck constant times c'),
    symbol='(N_A*h*c)',
    u_symbol='(N_A·h·c)'
)
h = Planck_constant = UnitConstant(
    'Planck_constant',
    _cd('Planck constant'),
    symbol='h'
)
hbar = Planck_constant_over_2_pi = UnitConstant(
    'Planck_constant_over_2_pi',
    _cd('Planck constant over 2 pi'),
    symbol='(h/(2*pi))',
    u_symbol='ħ'
)
quantum_of_circulation = UnitConstant(
    'quantum_of_circulation',
    _cd('quantum of circulation'),
    symbol='(h/(2*m_e))',
    u_symbol='(h/(2·mₑ))'
)
quantum_of_circulation_times_2 = UnitConstant(
    'quantum_of_circulation_times_2',
    _cd('quantum of circulation times 2'),
    symbol='(h/m_e)',
    u_symbol='(h/mₑ)'
)

l_P = Planck_length = UnitConstant(
    'Planck_length',
    _cd('Planck length'),
    symbol='l_P'
)
m_P = Planck_mass = UnitConstant(
    'Planck_mass',
    _cd('Planck mass'),
    symbol='m_P'
)
T_P = Planck_temperature = UnitConstant(
    'Planck_temperature',
    _cd('Planck temperature'),
    symbol='T_P'
)
t_P = Planck_time = UnitConstant(
    'Planck_time',
    _cd('Planck time'),
    symbol='t_P'
)

R_infinity = Rydberg_constant = UnitConstant(
    'Rydberg_constant',
    _cd('Rydberg constant'),
    symbol='R_infinity',
    u_symbol='R_∞'
)
Rydberg_constant_times_c_in_Hz = UnitConstant(
    'Rydberg_constant_times_c_in_Hz',
    _cd('Rydberg constant times c in Hz')
)
Rydberg_constant_times_hc_in_eV = UnitConstant(
    'Rydberg_constant_times_hc_in_eV',
    _cd('Rydberg constant times hc in eV')
)
Rydberg_constant_times_hc_in_J = UnitConstant(
    'Rydberg_constant_times_hc_in_J',
    _cd('Rydberg constant times hc in J'),
    symbol='(R_infinity*h*c)',
    u_symbol='(R_∞·h·c)'
)

G_0 = conductance_quantum = UnitConstant(
    'conductance_quantum',
    _cd('conductance quantum'),
    symbol='G_0',
    u_symbol='G₀'
)
K_J90 = conventional_value_of_Josephson_constant = UnitConstant(
    'conventional_value_of_Josephson_constant',
    _cd('conventional value of Josephson constant')
)
R_K90 = conventional_value_of_von_Klitzing_constant = UnitConstant(
    'conventional_value_of_von_Klitzing_constant',
    _cd('conventional value of von Klitzing constant')
)
Fermi_coupling_constant = UnitConstant(
    'Fermi_coupling_constant',
    _cd('Fermi coupling constant'),
    symbol='(G_F/(hbar*c)**3)',
    u_symbol='(G_F/(ħ·c)³)'
)
alpha = fine_structure_constant = UnitConstant(
    'fine_structure_constant',
    _cd('fine-structure constant'),
    symbol='alpha',
    u_symbol='α'
)
inverse_fine_structure_constant = UnitConstant(
    'inverse_fine_structure_constant',
    _cd('inverse fine-structure constant'),
    symbol='alpha**-1',
    u_symbol='α⁻¹'
)
c_1 = first_radiation_constant = UnitConstant(
    'first_radiation_constant',
    _cd('first radiation constant'),
    symbol='c_1',
    u_symbol='c₁'
)
c_1L = first_radiation_constant_for_spectral_radiance = UnitConstant(
    'first_radiation_constant_for_spectral_radiance',
    _cd('first radiation constant for spectral radiance'),
    symbol='c_1L',
    u_symbol='c₁_L'
)
inverse_of_conductance_quantum = UnitConstant(
    'inverse_of_conductance_quantum',
    _cd('inverse of conductance quantum'),
    symbol='G_0**-1',
    u_symbol='G₀⁻¹'
)
Josephson_constant = K_J = UnitConstant(
    'Josephson_constant',
    _cd('Josephson constant'),
    symbol='K_J'
)
Phi_0 = magnetic_flux_quantum = UnitConstant(
    'magnetic_flux_quantum',
    _cd('magnetic flux quantum'),
    symbol='Phi_0',
    u_symbol='Φ₀'
)
Newtonian_constant_of_gravitation_over_h_bar_c = UnitConstant(
    'Newtonian_constant_of_gravitation_over_h_bar_c',
    _cd('Newtonian constant of gravitation over h-bar c'),
    symbol='(G/(hbar*c))',
    u_symbol='(G/(ħ·c))'
)
Sackur_Tetrode_constant_ST_100kPa = UnitConstant(
    'Sackur_Tetrode_constant_ST_100kPa',
    _cd('Sackur-Tetrode constant (1 K, 100 kPa)')
)
Sackur_Tetrode_constant_STP = UnitConstant(
    'Sackur_Tetrode_constant_STP',
    _cd('Sackur-Tetrode constant (1 K, 101.325 kPa)')
)
c_2 = second_radiation_constant = UnitConstant(
    'second_radiation_constant',
    _cd('second radiation constant'),
    symbol='c_2',
    u_symbol='c₂'
)
sigma = Stefan_Boltzmann_constant = UnitConstant(
    'Stefan_Boltzmann_constant',
    _cd('Stefan-Boltzmann constant'),
    symbol='sigma',
    u_symbol='σ'
)
R_K = von_Klitzing_constant = UnitConstant(
    'von_Klitzing_constant',
    _cd('von Klitzing constant'),
    symbol='R_K'
)
b_prime = Wien_frequency_displacement_law_constant = UnitConstant(
    'Wien_frequency_displacement_law_constant',
    _cd('Wien frequency displacement law constant'),
    symbol='bprime',
    u_symbol='b′'
)
b = Wien_wavelength_displacement_law_constant = UnitConstant(
    'Wien_wavelength_displacement_law_constant',
    _cd('Wien wavelength displacement law constant'),
    symbol='b'
)

Planck_constant_in_eV_s = UnitConstant(
    'Planck_constant_in_eV_s',
    _cd('Planck constant in eV s')
)
Planck_constant_over_2_pi_in_eV_s = UnitConstant(
    'Planck_constant_over_2_pi_in_eV_s',
    _cd('Planck constant over 2 pi in eV s')
)
Planck_constant_over_2_pi_times_c_in_MeV_fm = UnitConstant(
    'Planck_constant_over_2_pi_times_c_in_MeV_fm',
    _cd('Planck constant over 2 pi times c in MeV fm')
)
Planck_mass_energy_equivalent_in_GeV = UnitConstant(
    'Planck_mass_energy_equivalent_in_GeV',
    _cd('Planck mass energy equivalent in GeV')
)

del UnitConstant, _cd
