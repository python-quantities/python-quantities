"""
"""
from __future__ import absolute_import

from ..unitquantity import dimensionless, UnitQuantity

percent = UnitQuantity(
    'percent',
    .01*dimensionless,
    symbol='%'
)

count = counts = UnitQuantity(
    'count',
    1*dimensionless,
    symbol='ct',
    aliases=['cts', 'counts']
)

lsb = UnitQuantity(
    'least_significant_bit',
    1*dimensionless,
    symbol='lsb',
    aliases=['lsbs']
)

del UnitQuantity
