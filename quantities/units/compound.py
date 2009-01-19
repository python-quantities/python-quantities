"""
"""
from __future__ import absolute_import

from ..markup import USE_UNICODE, superscript
from ..unitquantity import UnitQuantity
from ..registry import unit_registry

class CompoundUnit(UnitQuantity):

    def __new__(cls, name):
        return UnitQuantity.__new__(cls, name, unit_registry[name])

    def __repr__(self):
        return '1 %s'%self.name

    @property
    def name(self):
        if USE_UNICODE:
            return '(%s)'%(superscript(self._name))
        else:
            return '(%s)'%self._name
