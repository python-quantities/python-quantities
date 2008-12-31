"""
"""

from quantities.units.unitquantity import UnitQuantity, UnitTime


s = sec = second = seconds = UnitTime('s')
ms = millisecond = milliseconds = UnitTime('ms')
us = microsecond = microseconds = UnitTime('us')
ns = nanosecond = nanoseconds = UnitTime('ns')
ps = picosecond = picoseconds = UnitTime('ps')
fs = femtosecond = femtoseconds = UnitTime('fs')
attosecond = attoseconds = UnitTime('attosecond') # as is a keyword in python2.6
jiffy = jiffies = UnitTime('jiffy') # no kidding?

shake = shakes = UnitTime('shake')
sidereal_day = sidereal_days = UnitTime('sidereal_day')
sidereal_minute = sidereal_minutes = UnitTime('sidereal_minute')
sidereal_hour = sidereal_hours = UnitTime('sidereal_hour')
sidereal_second = sidereal_seconds = UnitTime('sidereal_second')
sidereal_year = sidereal_years = UnitTime('sidereal_year')
sidereal_month = sidereal_months = UnitTime('sidereal_month')
tropical_month = tropical_months = UnitTime('tropical_month')
lunar_month = lunar_months = UnitTime('lunar_month')
common_year = common_years = UnitTime('common_year')
leap_year = leap_years = UnitTime('leap_year')
Julian_year = Julian_years = UnitTime('Julian_year')
Gregorian_year = Gregorian_years = UnitTime('Gregorian_year')

h = hr = hour = hours = UnitTime('hr')
min = minute = minutes = UnitTime('min')
d = day = days = UnitTime('day')
week = weeks = UnitTime('week')
fortnight = UnitTime('fortnight')
yr = year = years = tropical_year = tropical_years = UnitTime('yr')
a = year # anno
eon = eons = UnitTime('eon')
month = UnitTime('month')

work_year = UnitQuantity('work_year', 2056*hours)
work_month = UnitQuantity('work_month', work_year/12)

del UnitQuantity
