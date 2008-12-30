"""
"""

from quantities.quantity import Quantity


class UnitQuantity(Quantity):

    def __new__(cls, name, *args, **kwargs):
        return Quantity.__new__(
            cls,
            1.0,
            dtype='d',
            dimensionality={},
            mutable=False
        )

    def __init__(self, name, *args, **kwargs):
        self._name = name
        Quantity.__init__(self, 1.0, 'd', {self:1}, mutable=False)

    @property
    def reference_quantity(self):
        return self

    @property
    def units(self):
        try:
            return self._name
        except:
            return ''

    def __repr__(self):
        return self.units

    __str__ = __repr__


class ReferenceUnit(UnitQuantity):
    pass


class CompoundUnit(UnitQuantity):

    def __init__(self, name, reference_quantity):
        UnitQuantity.__init__(self, name)
        self._reference_quantity = reference_quantity

    @property
    def reference_quantity(self):
        return self._reference_quantity


m = ReferenceUnit('m')
kg = ReferenceUnit('kg')
s = ReferenceUnit('s')
J = CompoundUnit('J', kg*m**2/s**2)

energy = J*J
