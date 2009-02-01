# -*- coding: utf-8 -*-

import math as _math

from ..unitquantity import UnitConstant


pi = UnitConstant(
    'pi',
    _math.pi,
    u_symbol='π'
)
golden = golden_ratio = UnitConstant(
    'golden_ratio',
    (1 + _math.sqrt(5)) / 2,
    u_symbol='ϕ',
    aliases='golden'
)
