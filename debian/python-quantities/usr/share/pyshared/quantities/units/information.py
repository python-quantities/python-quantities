"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity, UnitInformation, dimensionless
from .time import s

bit = UnitInformation(
    'bit',
    aliases=['bits']
)
B = byte = o = octet = UnitInformation(
    'byte',
    8*bit,
    symbol='B',
    aliases=['bytes', 'o', 'octet', 'octets']
)
Bd = baud = bps = UnitQuantity(
    'baud',
    bit/s,
    symbol='Bd',
)

del UnitQuantity, s, dimensionless
