"""
"""

from quantities.units.unitquantity import UnitAngle, UnitQuantity

turn = circle = UnitAngle('turn')
radian = radians = UnitAngle('radian')
arcdeg = arcdegs = degree = degrees = angular_degree = angular_degrees = \
    UnitAngle('arcdeg')
arcmin = arcmins = arcminute = arcminutes = angular_minute = \
    angular_minutes = UnitAngle('arcmin')
arcsec = arcsecs = arcseconds = arcseconds = angular_second = \
    angular_seconds = UnitAngle('arcsec')
grade = grades = UnitAngle('grade')

degree_north = degrees_north = degree_N = degrees_N = UnitAngle('degrees_N')
degree_east = degrees_east = degree_E= degrees_E = UnitAngle('degrees_E')
degree_west = degrees_west = degree_W= degrees_W = UnitAngle('degrees_W')
degree_true = degrees_true = degree_T = degrees_T = UnitAngle('degrees_T')

sr = steradian = steradians = UnitQuantity('sr')

del UnitQuantity
