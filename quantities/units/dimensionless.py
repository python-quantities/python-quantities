"""
"""

from quantities.unitquantity import dimensionless, UnitQuantity

percent = UnitQuantity(
    'percent',
    .01*dimensionless,
    symbol='%'
)

del UnitQuantity
