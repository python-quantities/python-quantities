"""
"""
from __future__ import absolute_import

from ..unitquantity import dimensionless, UnitQuantity

percent = UnitQuantity(
    'percent',
    .01*dimensionless,
    symbol='%'
)

del UnitQuantity
