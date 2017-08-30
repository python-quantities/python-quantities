# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity, UnitTime


s = sec = second = UnitTime(
    'second',
    symbol='s',
    aliases=['sec', 'seconds']
)
ks = kilosecond = UnitTime(
    'kilosecond',
    s*1000,
    'ks',
    aliases=['kiloseconds']
)
Ms = megasecond = UnitTime(
    'megasecond',
    ks*1000,
    'Ms',
    aliases=['megaseconds']
)
ms = millisecond = UnitTime(
    'millisecond',
    s/1000,
    'ms',
    aliases=['milliseconds']
)
us = microsecond = UnitTime(
    'microsecond',
    ms/1000,
    symbol='us',
    u_symbol='µs',
    aliases=['microseconds']
)
ns = nanosecond = UnitTime(
    'nanosecond',
    us/1000,
    symbol='ns',
    aliases=['nanoseconds']
)
ps = picosecond = UnitTime(
    'picosecond',
    ns/1000,
    symbol='ps',
    aliases=['picoseconds']
)
fs = femtosecond = UnitTime(
    'femtosecond',
    ps/1000,
    symbol='fs',
    aliases=['femtoseconds']
)
attosecond = UnitTime(
    'attosecond',
    fs/1000,
    symbol='as',
    aliases=['attoseconds']
) # as is a keyword in python2.6

min = minute = UnitTime(
    'minute',
    60*s,
    symbol='min',
    aliases=['minutes']
)
h = hr = hour = UnitTime(
    'hour',
    60*min,
    symbol='h',
    aliases=['hr', 'hours']
)
d = day = UnitTime(
    'day',
    24*hr,
    symbol='d',
    aliases=['days']
)
week = UnitTime(
    'week',
    7*day,
    aliases=['weeks']
)
fortnight = UnitTime(
    'fortnight',
    2*week,
    aliases=['fortnights']
)
yr = year = tropical_year = a = UnitTime(
    'year',
    31556925.9747*s,
    symbol='yr',
    aliases=['a', 'years', 'tropical_year', 'tropical_years'],
    doc='a is an acceptable alias for year, short for anno'
)
month = UnitTime(
    'month',
    yr/12,
    aliases=['months']
)
shake = UnitTime(
    'shake',
    1e-8*s,
    aliases=['shakes']
)

sidereal_day = UnitTime(
    'sidereal_day',
    day/1.00273790935079524,
    aliases=['sidereal_days'],
    doc='''
    approximate.

    http://en.wikipedia.org/wiki/Sidereal_time
    '''
)
sidereal_hour = UnitTime(
    'sidereal_hour',
    sidereal_day/24,
    aliases=['sidereal_hours']
)
sidereal_minute = UnitTime(
    'sidereal_minute',
    sidereal_hour/60,
    aliases=['sidereal_minutes']
)
sidereal_second = UnitTime(
    'sidereal_second',
    sidereal_minute/60,
    aliases=['sidereal_seconds']
)
sidereal_year = UnitTime(
    'sidereal_year',
    366.25636042*sidereal_day,
    aliases=['sidereal_years'],
    doc='http://en.wikipedia.org/wiki/Sidereal_year'
)
sidereal_month = UnitTime(
    'sidereal_month',
    27.321661*day,
    aliases=['sidereal_months'],
    doc='http://en.wikipedia.org/wiki/Month#Sidereal_month'
)

tropical_month = UnitTime(
    'tropical_month',
    27.321582*day,
    aliases=['tropical_months']
)
synodic_month = lunar_month = UnitTime(
    'synodic_month',
    29.530589*day,
    aliases=['synodic_months', 'lunar_month', 'lunar_months'],
    doc='''
    long-term average. 

    http://en.wikipedia.org/wiki/Month#Synodic_month
    '''
)
common_year = UnitTime(
    'common_year',
    365*day,
    aliases=['common_years']
)
leap_year = UnitTime(
    'leap_year',
    366*day,
    aliases=['leap_years']
)
Julian_year = UnitTime(
    'Julian_year',
    365.25*day,
    aliases=['Julian_years']
)
Gregorian_year = UnitTime(
    'Gregorian_year',
    365.2425*day,
    aliases=['Gregorian_years']
)

millenium = UnitTime(
    'millenium',
    1000*year,
    aliases=['millenia']
)
eon = UnitTime(
    'eon',
    1e9*year,
    aliases=['eons']
)

work_year = UnitQuantity(
    'work_year',
    2056*hour,
    aliases=['work_years']
)
work_month = UnitQuantity(
    'work_month',
    work_year/12,
    aliases=['work_months']
)

del UnitQuantity
