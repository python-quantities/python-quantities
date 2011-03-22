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
