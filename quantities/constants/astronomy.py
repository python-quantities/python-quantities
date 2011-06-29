# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..uncertainquantity import UncertainQuantity
from ..unitquantity import UnitConstant


au = astronomical_unit = UnitConstant(
    'astronomical_unit',
    UncertainQuantity(149597870700, 'm', 3),
    symbol='au',
    doc='http://en.wikipedia.org/wiki/Astronomical_unit'
)
G = Newtonian_constant_of_gravitation = UnitConstant(
    'Newtonian_constant_of_gravitation',
    _cd('Newtonian constant of gravitation'),
    symbol='G'
)

del UnitConstant, UncertainQuantity, _cd
