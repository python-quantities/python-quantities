# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

#import math as _math
from .utils import _cd
from quantities.unitquantity import UnitConstant


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

del UnitConstant, _cd
