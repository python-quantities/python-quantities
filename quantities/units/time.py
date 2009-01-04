"""
"""

from quantities.units.unitquantity import UnitQuantity, UnitTime


s = sec = second = seconds = \
    UnitTime('s')
ms = millisecond = milliseconds = \
    UnitTime('ms', s/1000)
us = microsecond = microseconds = \
    UnitTime('us', ms/1000)
ns = nanosecond = nanoseconds = \
    UnitTime('ns', us/1000)
ps = picosecond = picoseconds = \
    UnitTime('ps', ns/1000)
fs = femtosecond = femtoseconds = \
    UnitTime('fs', ps/1000)
attosecond = attoseconds = \
    UnitTime('attosecond', fs/1000) # as is a keyword in python2.6
jiffy = jiffies = \
    UnitTime('jiffy', s/100) # no kidding?

min = minute = minutes = \
    UnitTime('min', 60*s)
h = hr = hour = hours = \
    UnitTime('hr', 60*min)
d = day = days = \
    UnitTime('day', 24*hr)
week = weeks = \
    UnitTime('week', 7*day)
fortnight = \
    UnitTime('fortnight', 2*weeks)
yr = year = years = tropical_year = tropical_years = a = \
    UnitTime('yr', 3.15569259747e7*s) # a for anno
month = \
    UnitTime('month', yr/12)
shake = shakes = \
    UnitTime('shake', 1e-8*s)
sidereal_second = sidereal_seconds = \
    UnitTime('sidereal_second', 0.9972696*s)
sidereal_minute = sidereal_minutes = \
    UnitTime('sidereal_minute', 60*sidereal_second)
sidereal_hour = sidereal_hours = \
    UnitTime('sidereal_hour', 60*sidereal_minute)
sidereal_day = sidereal_days = \
    UnitTime('sidereal_day', 24*sidereal_hour)
sidereal_year = sidereal_years = \
    UnitTime('sidereal_year', 1.00003878*year)
sidereal_month = sidereal_months = \
    UnitTime('sidereal_month', 27.321661*day)
tropical_month = tropical_months = \
    UnitTime('tropical_month', 27.321582*day)
lunar_month = lunar_months = \
    UnitTime('lunar_month', 29.530589*day)
common_year = common_years = \
    UnitTime('common_year', 365*day)
leap_year = leap_years = \
    UnitTime('leap_year', 366*day)
Julian_year = Julian_years = \
    UnitTime('Julian_year', 365.25*day)
Gregorian_year = Gregorian_years = \
    UnitTime('Gregorian_year', 365.2425*day)

millenium = millenia = \
    UnitTime('eon', 1000*year)
eon = eons = \
    UnitTime('eon', 1e9*year)

work_year = \
    UnitQuantity('work_year', 2056*hours)
work_month = \
    UnitQuantity('work_month', work_year/12)

del UnitQuantity
