"""
"""

from quantities.units.unitquantities import compound, time


s = sec = second = seconds = time('s')
ms = millisecond = milliseconds = time('ms')
us = microsecond = microseconds = time('us')
ns = nanosecond = nanoseconds = time('ns')
ps = picosecond = picoseconds = time('ps')
fs = femtosecond = femtoseconds = time('fs')
as = attosecond = attoseconds = time('as')
jiffy = jiffies = time('jiffy') # no kidding?

shake = shakes = time('shake')
sidereal_day = sidereal_days = time('sidereal_day')
sidereal_minute = sidereal_minutes = time('sidereal_minute')
sidereal_hour = sidereal_hours = time('sidereal_hour')
sidereal_second = sidereal_seconds = time('sidereal_second')
sidereal_year = sidereal_years = time('sidereal_year')
sidereal_month = sidereal_months = time('sidereal_month')
tropical_month = tropical_months = time('tropical_month')
lunar_month = lunar_months = time('lunar_month')
common_year = common_years = time('common_year')
leap_year = leap_years = time('leap_year')
Julian_year = Julian_years = time('Julian_year')
Gregorian_year = Gregorian_years = time('Gregorian_year')

h = hr = hour = hours = time('hr')
min = minute = minutes = time('min')
d = day = days = time('day')
week = weeks = time('week')
fortnight = time('fortnight')
yr = year = years = tropical_year = tropical_years = time('yr')
a = year # anno
eon = eons = time('eon')
month = time('month')

work_year = 2056 * hours
work_month = work_year/12

work_year_ = compound('work_year')
work_month_ = compound('work_month')

del compound
