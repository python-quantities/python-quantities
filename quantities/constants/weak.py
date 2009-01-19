# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ._utils import _cd
from ..unitquantity import UnitConstant


weak_mixing_angle = UnitConstant(
    'weak_mixing_angle',
    _cd('weak mixing angle')
)

del UnitConstant, _cd
