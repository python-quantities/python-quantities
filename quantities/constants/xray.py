# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


Angstrom_star = UnitConstant(
    'Angstrom_star',
    _cd('Angstrom star'),
    symbol='A*',
    u_symbol='Å*'
)
d_220 = a_Si_220 = silicon_220_lattice_spacing = UnitConstant(
    'silicon_220_lattice_spacing',
    _cd('{220} lattice spacing of silicon'),
    symbol='d_220',
    u_symbol='d₂₂₀'
)
Cu_x_unit = UnitConstant(
    'Cu_x_unit',
    _cd('Cu x unit'),
    symbol='CuKalpha_1',
    u_symbol='CuKα₁'
)
a = a_Si_100 = lattice_parameter_of_silicon = UnitConstant(
    'lattice_parameter_of_silicon',
    _cd('lattice parameter of silicon')
)
Mo_x_unit = UnitConstant(
    'Mo_x_unit',
    _cd('Mo x unit'),
    symbol='MoKalpha_1',
    u_symbol='MoKα₁'
)

del UnitConstant, _cd
