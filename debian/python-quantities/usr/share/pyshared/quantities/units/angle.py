# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from math import pi

from ..unitquantity import UnitQuantity, dimensionless

rad = radian = radians = UnitQuantity(
    'radian',
    1*dimensionless,
    symbol='rad',
    aliases=['radians']
)
mrad = milliradian = UnitQuantity(
    'milliradian',
    rad/1000,
    symbol='mrad',
    aliases=['milliradians']
)
urad = microradian = UnitQuantity(
    'microradian',
    mrad/1000,
    symbol='urad',
    u_symbol='µrad',
    aliases=['microradians']
)

turn = revolution = cycle = turns = circle = circles = UnitQuantity(
    'turn',
    2*pi*radian,
    aliases=['turns', 'revolutions', 'circles', 'cycles']
)
deg = degree = degrees = arcdeg = arcdegree = angular_degree = UnitQuantity(
    'arcdegree',
    pi/180*radian,
    symbol='deg',
    u_symbol='°',
    aliases=[
        'degree', 'degrees', 'arc_degree', 'arc_degrees', 'angular_degree',
        'angular_degrees', 'arcdegrees', 'arcdeg'
    ]
)
arcminute = arcmin = arc_minute = angular_minute = UnitQuantity(
    'arcminute',
    arcdeg/60,
    symbol='arcmin',
    u_symbol='′',
    aliases=[
        'arcmins', 'arcminutes', 'arc_minute', 'arc_minutes',
        'angular_minute', 'angular_minutes'
    ]
)
arcsecond = arcsec = arc_second = angular_second = UnitQuantity(
    'arcsecond',
    arcmin/60,
    symbol='arcsec',
    u_symbol='″',
    aliases=[
        'arcsecs', 'arcseconds', 'arc_second', 'arc_seconds',
        'angular_second', 'angular_seconds'
    ]
)
grad = grade = UnitQuantity(
    'grad',
    0.9*arcdeg,
    aliases=['grads', 'grade', 'grades', 'gron', 'grons', 'gradian', 'gradians']
)

degrees_north = degrees_N = UnitQuantity(
    'degrees_north',
    arcdeg,
    symbol='degN',
    u_symbol='°N',
    aliases=['degrees_N']
)
degrees_east = degrees_E = UnitQuantity(
    'degrees_east',
    arcdeg,
    symbol='degE',
    u_symbol='°E',
    aliases=['degrees_E']
)
degrees_west = degrees_W = UnitQuantity(
    'degrees_west',
    arcdeg,
    symbol='degW',
    u_symbol='°W',
    aliases=['degrees_W']
)
degrees_true = degrees_T = UnitQuantity(
    'degrees_true',
    arcdeg,
    symbol='degT',
    u_symbol='°T',
    aliases=['degrees_T']
)

sr = steradian = UnitQuantity(
    'steradian',
    radian**2,
    symbol='sr',
    aliases=['steradians']
)

del UnitQuantity
