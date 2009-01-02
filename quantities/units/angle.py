"""
"""

from numpy import pi

from quantities.units.unitquantity import UnitAngle, UnitQuantity

radian = radians = \
    UnitAngle('radian')
turn = turns = circle = circles = \
    UnitAngle('turn', 2*pi*radian)
arcdeg = arcdegs = degree = degrees = angular_degree = angular_degrees = \
    UnitAngle('arcdeg', pi/180*radian)
arcmin = arcmins = arcminute = arcminutes = angular_minute = \
    angular_minutes = \
    UnitAngle('arcmin', arcdeg/60)
arcsec = arcsecs = arcseconds = arcseconds = angular_second = \
    angular_seconds = \
    UnitAngle('arcsec', arcmin/60)
grade = grades = \
    UnitAngle('grade', 0.9*arcdeg)

degree_north = degrees_north = degree_N = degrees_N = \
    UnitAngle('degrees_N', arcdeg)
degree_east = degrees_east = degree_E= degrees_E = \
    UnitAngle('degrees_E', arcdeg)
degree_west = degrees_west = degree_W= degrees_W = \
    UnitAngle('degrees_W', arcdeg)
degree_true = degrees_true = degree_T = degrees_T = \
    UnitAngle('degrees_T', arcdeg)

sr = steradian = steradians = \
    UnitQuantity('sr', radian**2)

del UnitQuantity
