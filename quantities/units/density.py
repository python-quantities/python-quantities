# -*- coding:utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .length import m, cm
from .mass import kg, g

gram_per_cubic_centimetre = UnitQuantity(
    'gram_per_cubic_centimetre',
    g / cm**3,
    aliases=['grams_per_cubic_centimetre']
)

kilogram_per_cubic_metre = UnitQuantity(
    'kilogram_per_cubic_metre',
    kg / m**3,
    aliases=['kilograms_per_cubic_metre']
)

