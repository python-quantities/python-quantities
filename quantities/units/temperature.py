# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitTemperature


K = degK = kelvin = Kelvin = UnitTemperature(
    'Kelvin',
    symbol='K',
    aliases=['degK', 'kelvin']
)
for prefix, symbolprefix, magnitude in (
        ('yotta', 'Y', 1e24),
        ('zetta', 'Z', 1e21),
        ('exa', 'E', 1e18),
        ('peta', 'P', 1e15),
        ('tera', 'T', 1e12),
        ('giga', 'G', 1e9),
        ('mega', 'M', 1e6),
        ('kilo', 'k', 1e3),
        ('hecto', 'h', 1e2),
        ('deka', 'da', 1e1),
        ('deci', 'd', 1e-1),
        ('centi', 'c', 1e-2),
        ('milli', 'm', 1e-3),
        ('micro', 'u', 1e-6),
        ('nano', 'n', 1e-9),
        ('pico', 'p', 1e-12),
        ('femto', 'f', 1e-15),
        ('atto', 'a', 1e-18),
        ('zepto', 'z', 1e-21),
        ('yocto', 'y', 1e-24),
):
    symbol = symbolprefix +'K'
    globals()[symbol] = UnitTemperature(
        prefix + 'kelvin',
        K*magnitude,
        symbol=symbol
    )

degR = rankine = Rankine = UnitTemperature(
    'Rankine',
    K/1.8,
    symbol='degR',
    u_symbol='°R',
    aliases=['rankine']
)
degC = celsius = Celsius = UnitTemperature(
    'Celsius',
    K,
    symbol='degC',
    u_symbol='°C',
    aliases=['celsius'],
    doc='''
    Unicode has special compatibility characters for ℃, but its use is
    discouraged by the unicode consortium.
    '''
)
degF = fahrenheit = Fahrenheit = UnitTemperature(
    'Fahrenheit',
    degR,
    symbol='degF',
    u_symbol='°F',
    aliases=['fahrenheit'],
    doc='''
    Unicode has special compatibility characters for ℉, but its use is
    discouraged by the unicode consortium.
    '''
)
