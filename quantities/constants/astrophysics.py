# -*- coding: utf-8 -*-
"""
"""

from quantities.constants._utils import _cd
from quantities.uncertainquantity import UncertainQuantity
from quantities.unitquantity import UnitConstant


au = astronomical_unit = UnitConstant(
    'astronomical_unit',
    UncertainQuantity(149597870691, 'm', 30),
    symbol='au'
) # http://en.wikipedia.org/wiki/Astronomical_unit
G = Newtonian_constant_of_gravitation = UnitConstant(
    'Newtonian_constant_of_gravitation',
    _cd('Newtonian constant of gravitation'),
    symbol='G'
)

del UnitConstant, UncertainQuantity, _cd
