# -*- coding: utf-8 -*-
"""
"""

from quantities.constants._utils import _cd
from quantities.unitquantity import UnitConstant


weak_mixing_angle = UnitConstant(
    'weak_mixing_angle',
    _cd('weak mixing angle')
)

del UnitConstant, _cd
