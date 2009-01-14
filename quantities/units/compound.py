"""
"""

from quantities.dimensionality import superscript
from quantities.units.unitquantity import UnitQuantity
from quantities.registry import unit_registry

class CompoundUnit(UnitQuantity):

    def __init__(self, name):
        UnitQuantity.__init__(self, '(%s)'%name, unit_registry[name])

    def __repr__(self):
        return '1 %s'%(superscript(self.name))
