"""
"""

from quantities.units.unitquantity import dimensionless, UnitQuantity

percent = UnitQuantity('percent', .01*dimensionless)

del UnitQuantity
