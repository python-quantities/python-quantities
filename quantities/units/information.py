"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity, UnitInformation, dimensionless
from .time import s

bit = UnitInformation(
    'bit',
    1*dimensionless,
    aliases=['bits']
)
B = byte = o = octet = UnitInformation(
    'byte',
    8*bit,
    symbol='B',
    aliases=['bytes', 'o', 'octet', 'octets']
)
count = counts = UnitInformation(
    'count',
    1*dimensionless,
    symbol='ct',
    aliases=['cts', 'counts']
)

Bd = baud = bps = UnitQuantity(
    'baud',
    bit/s,
    symbol='Bd',
)
cps = UnitQuantity(
    'counts_per_second',
    count/s
)

del UnitQuantity, s, dimensionless
