# -*- coding: utf-8 -*-
"""
"""

from quantities.constants.utils import _cd
from quantities.quantity import Quantity
from quantities.uncertainquantity import UncertainQuantity
from quantities.unitquantity import UnitConstant


d_220 = a_Si_220 = silicon_220_lattice_spacing = UnitConstant(
    'silicon_220_lattice_spacing',
    _cd('{220} lattice spacing of silicon'),
    symbol='d_220',
    u_symbol='d₂₂₀'
)
m_alpha = alpha_particle_mass = UnitConstant(
    'alpha_particle_mass',
    _cd('alpha particle mass'),
    symbol='m_alpha',
    u_symbol='m_α'
)
Angstrom_star = UnitConstant(
    'Angstrom_star',
    _cd('Angstrom star'),
    symbol='A*',
    u_symbol='Å*'
)
au = astronomical_unit = UnitConstant(
    'astronomical_unit',
    UncertainQuantity(149597870691, 'm', 30),
    symbol='au'
) # http://en.wikipedia.org/wiki/Astronomical_unit
amu = atomic_mass_constant = UnitConstant(
    'atomic_mass_constant',
    _cd('atomic mass constant'),
    symbol='m_u',
    u_symbol='mᵤ'
)
atomic_unit_of_1st_hyperpolarizablity = UnitConstant(
    'atomic_unit_of_1st_hyperpolarizablity',
    _cd('atomic unit of 1st hyperpolarizablity'),
    symbol='(e**3*a_0**3/E_h**2)',
    u_symbol='(e³·a₀³/E_h²)'
)
atomic_unit_of_2nd_hyperpolarizablity = UnitConstant(
    'atomic_unit_of_2nd_hyperpolarizablity',
    _cd('atomic unit of 2nd hyperpolarizablity'),
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
atomic_unit_of_electric_polarizablity = UnitConstant(
    'atomic_unit_of_electric_polarizablity',
    _cd('atomic unit of electric polarizablity'),
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
N_A = L = Avogadro_constant = UnitConstant(
    'Avogadro_constant',
    _cd('Avogadro constant'),
    symbol='N_A'
)
mu_B = Bohr_magneton = UnitConstant(
    'Bohr_magneton',
    _cd('Bohr magneton'),
    symbol='mu_B',
    u_symbol='μ_B'
)
a_0 = Bohr_radius = UnitConstant(
    'Bohr_radius',
    _cd('Bohr radius')
)
k = Boltzmann_constant = UnitConstant(
    'Boltzmann_constant',
    _cd('Boltzmann constant'),
    symbol='k'
)
Z_0 = characteristic_impedance_of_vacuum = UnitConstant(
    'characteristic_impedance_of_vacuum',
    _cd('characteristic impedance of vacuum'),
    symbol='Z_0',
    u_symbol='Z₀'
)
r_e = classical_electron_radius = UnitConstant(
    'classical_electron_radius',
    _cd('classical electron radius'),
    symbol='r_e',
    u_symbol='rₑ'
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
Cu_x_unit = UnitConstant(
    'Cu_x_unit',
    _cd('Cu x unit'),
    symbol='CuKalpha_1',
    u_symbol='CuKα₁'
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
m_d = deuteron_mass = UnitConstant(
    'deuteron_mass',
    _cd('deuteron mass'),
    symbol='m_d'
)
R_d = deuteron_rms_charge_radius = UnitConstant(
    'deuteron_rms_charge_radius',
    _cd('deuteron rms charge radius'),
    symbol='R_d'
)
vacuum_permittivity = epsilon_0 = electric_constant = UnitConstant(
    'electric_constant',
    _cd('electric constant'),
    symbol='epsilon_0',
    u_symbol='ε₀'
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
m_e = electron_mass = UnitConstant(
    'electron_mass',
    _cd('electron mass'),
    symbol='m_e',
    u_symbol='mₑ'
)
eV = electron_volt = UnitConstant(
    'electron_volt',
    _cd('electron volt'),
    symbol='eV'
)
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
F = Faraday_constant = UnitConstant(
    'Faraday_constant',
    _cd('Faraday constant'),
    symbol='F'
)
#F_star = Faraday_constant_for_conventional_electric_current = UnitConstant(
#    _cd('Faraday constant for conventional electric current') what is a unit of C_90?
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
E_h = Hartree_energy = UnitConstant(
    'Hartree_energy',
    _cd('Hartree energy'),
    symbol='E_h'
)
m_h = helion_mass = UnitConstant(
    'helion_mass',
    _cd('helion mass'),
    symbol='m_h'
)
inverse_fine_structure_constant = UnitConstant(
    'inverse_fine_structure_constant',
    _cd('inverse fine-structure constant'),
    symbol='alpha**-1',
    u_symbol='α⁻¹'
)
inverse_of_conductance_quantum = UnitConstant(
    'inverse_of_conductance_quantum',
    _cd('inverse of conductance quantum'),
    symbol='G_0**-1',
    u_symbol='G₀⁻¹'
)
Josephson_constant = UnitConstant(
    'Josephson_constant',
    _cd('Josephson constant'),
    symbol='K_J'
)
a = a_Si_100 = lattice_parameter_of_silicon = UnitConstant(
    'lattice_parameter_of_silicon',
    _cd('lattice parameter of silicon')
)
n_0 = Loschmidt_constant = UnitConstant(
    'Loschmidt_constant',
    _cd('Loschmidt constant (273.15 K, 101.325 kPa)'),
    symbol='n_0',
    u_symbol='n₀'
)
mu_0 = magnetic_constant = UnitConstant(
    'magnetic_constant',
    _cd('magnetic constant'),
    symbol='mu_0',
    u_symbol='μ₀'
)
Phi_0 = magnetic_flux_quantum = UnitConstant(
    'magnetic_flux_quantum',
    _cd('magnetic flux quantum'),
    symbol='Phi_0',
    u_symbol='Φ₀'
)
R = molar_gas_constant = UnitConstant(
    'molar_gas_constant',
    _cd('molar gas constant'),
    symbol='R'
)
M_u = molar_mass_constant = UnitConstant(
    'molar_mass_constant',
    _cd('molar mass constant'),
    symbol='M_u',
    u_symbol='Mᵤ'
)
molar_mass_of_carbon_12 = UnitConstant(
    'molar_mass_of_carbon_12',
    _cd('molar mass of carbon-12'),
    symbol='M_C12',
    u_symbol='M_¹²C'
)
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
molar_volume_of_ideal_gas_ST_100kPa = UnitConstant(
    'molar_volume_of_ideal_gas_ST_100kPa',
    _cd('molar volume of ideal gas (273.15 K, 100 kPa)')
)
molar_volume_of_ideal_gas_STP = UnitConstant(
    'molar_volume_of_ideal_gas_STP',
    _cd('molar volume of ideal gas (273.15 K, 101.325 kPa)')
)
molar_volume_of_silicon = UnitConstant(
    'molar_volume_of_silicon',
    _cd('molar volume of silicon')
)
Mo_x_unit = UnitConstant(
    'Mo_x_unit',
    _cd('Mo x unit'),
    symbol='MoKalpha_1',
    u_symbol='MoKα₁'
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
m_mu = muon_mass = UnitConstant(
    'muon_mass',
    _cd('muon mass'),
    symbol='m_mu',
    u_symbol='m_μ'
)
natural_unit_of_action = UnitConstant(
    'natural_unit_of_action',
    _cd('natural unit of action'),
    symbol='hbar',
    u_symbol='ħ'
)
natural_unit_of_energy = UnitConstant(
    'natural_unit_of_energy',
    _cd('natural unit of energy'),
    symbol='(m_e*c**2)',
    u_symbol='(mₑ·c²)'
)
natural_unit_of_length = UnitConstant(
    'natural_unit_of_length',
    _cd('natural unit of length'),
    symbol='lambdabar_C',
    u_symbol='ƛ_C'
)
natural_unit_of_mass = UnitConstant(
    'natural_unit_of_mass',
    _cd('natural unit of mass'),
    symbol='m_e',
    u_symbol='mₑ'
)
natural_unit_of_momentum = UnitConstant(
    'natural_unit_of_momentum',
    _cd('natural unit of momentum'),
    symbol='(m_e*c)',
    u_symbol='(mₑ·c)'
)
natural_unit_of_time = UnitConstant(
    'natural_unit_of_time',
    _cd('natural unit of time'),
    symbol='(hbar/(m_e*c**2))',
    u_symbol='(ħ/(mₑ·c²))'
)
natural_unit_of_velocity = UnitConstant(
    'natural_unit_of_velocity',
    _cd('natural unit of velocity'),
    symbol='c'
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
m_n = neutron_mass = UnitConstant(
    'neutron_mass',
    _cd('neutron mass'),
    symbol='m_n'
)
G = Newtonian_constant_of_gravitation = UnitConstant(
    'Newtonian_constant_of_gravitation',
    _cd('Newtonian constant of gravitation'),
    symbol='G'
)
Newtonian_constant_of_gravitation_over_h_bar_c = UnitConstant(
    'Newtonian_constant_of_gravitation_over_h_bar_c',
    _cd('Newtonian constant of gravitation over h-bar c'),
    symbol='(G/(hbar*c))',
    u_symbol='(G/(ħ·c))'
)
mu_N = nuclear_magneton = UnitConstant(
    'nuclear_magneton',
    _cd('nuclear magneton'),
    symbol='mu_N',
    u_symbol='μ_N'
)
h = Planck_constant = UnitConstant(
    'Planck_constant',
    _cd('Planck constant'),
    symbol='h'
)
Planck_constant_over_2_pi = UnitConstant(
    'Planck_constant_over_2_pi',
    _cd('Planck constant over 2 pi'),
    symbol='(h/(2*pi))',
    u_symbol='ħ'
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
m_p = proton_mass = UnitConstant(
    'proton_mass',
    _cd('proton mass'),
    symbol='m_p'
)
R_p = proton_rms_charge_radius = UnitConstant(
    'proton_rms_charge_radius',
    _cd('proton rms charge radius'),
    symbol='R_p'
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
R_infinity = Rydberg_constant = UnitConstant(
    'Rydberg_constant',
    _cd('Rydberg constant'),
    symbol='R_infinity',
    u_symbol='R_∞'
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
sigma = Stefan_Boltzmann_constant = UnitConstant(
    'Stefan_Boltzmann_constant',
    _cd('Stefan-Boltzmann constant'),
    symbol='sigma',
    u_symbol='σ'
)
m_tau = tau_mass = UnitConstant(
    'tau_mass',
    _cd('tau mass'),
    symbol='m_tau',
    u_symbol='m_τ'
)
sigma_e = Thomson_cross_section = UnitConstant(
    'Thomson_cross_section',
    _cd('Thomson cross section'),
    symbol='sigma_e',
    u_symbol='σₑ'
)
g_t = triton_g_factor = UnitConstant(
    'triton_g_factor',
    _cd('triton g factor'),
    symbol='g_t'
)
mu_t = triton_magnetic_moment = UnitConstant(
    'triton_magnetic_moment',
    _cd('triton magnetic moment'),
    symbol='mu_t',
    u_symbol='μ_t'
)
m_t = triton_mass = UnitConstant(
    'triton_mass',
    _cd('triton mass'),
    symbol='m_t'
)
u = unified_atomic_mass_unit = UnitConstant(
    'unified_atomic_mass_unit',
    _cd('unified atomic mass unit'),
    symbol='u'
)
R_K = von_Klitzing_constant = UnitConstant(
    'von_Klitzing_constant',
    _cd('von Klitzing constant'),
    symbol='R_K'
)
weak_mixing_angle = UnitConstant(
    'weak_mixing_angle',
    _cd('weak mixing angle')
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


del UnitConstant, Quantity, UncertainQuantity, _cd
