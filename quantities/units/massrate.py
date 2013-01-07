# -*- coding:utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity
from .time import s, yr
from .mass import g, solar_mass

gram_per_second = UnitQuantity(
    'gram_per_second',
    g / s,
    aliases=['grams_per_second']
)

solar_mass_per_year = UnitQuantity(
    'solar_mass_per_year',
    solar_mass / yr,
    aliases=['solar_masses_per_year']
)

del UnitQuantity, s, yr, g, solar_mass
