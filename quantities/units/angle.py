"""
"""

from quantities.units.unitquantities import angle, compound

turn = circle = angle('turn')
radian = radians = angle('radian')
arcdeg = arcdegs = degree = degrees = angular_degree = angular_degrees = \
    angle('arcdeg')
arcmin = arcmins = arcminute = arcminutes = angular_minute = \
    angular_minutes = angle('arcmin')
arcsec = arcsecs = arcseconds = arcseconds = angular_second = \
    angular_seconds = angle('arcsec')
grade = grades = angle('grade')

degree_north = degrees_north = degree_N = degrees_N = angle('degrees_N')
degree_east = degrees_east = degree_E= degrees_E = angle('degrees_E')
degree_west = degrees_west = degree_W= degrees_W = angle('degrees_W')
degree_true = degrees_true = degree_T = degrees_T = angle('degrees_T')

sr = steradian = steradians = radian**2

del compound
