"""
"""

from quantities.quantity import Quantity


class UnitQuantity(Quantity):

    def __new__(cls, name, reference_quantity=None):
        return Quantity.__new__(
            cls,
            1.0,
            dtype='d',
            dimensionality={},
            mutable=False
        )

    def __init__(self, name, reference_quantity=None):
        self._name = name
        Quantity.__init__(self, 1.0, 'd', {self:1}, mutable=False)

        if reference_quantity is None:
            reference_quantity = self
        self._reference_quantity = reference_quantity

    @property
    def reference_quantity(self):
        return self._reference_quantity

    @property
    def units(self):
        return self._name

    def __repr__(self):
        return self.units

    __str__ = __repr__


class UnitMass(UnitQuantity):
    pass


class UnitLength(UnitQuantity):
    pass


class UnitTime(UnitQuantity):
    pass


class UnitCharge(UnitQuantity):
    pass


class UnitLuminousIntensity(UnitQuantity):
    pass


class UnitSubstance(UnitQuantity):
    pass


class UnitTemperature(UnitQuantity):
    pass


class UnitInformation(UnitQuantity):
    pass


class UnitAngle(UnitQuantity):
    pass


class UnitCurrency(UnitQuantity):
    pass


class CompoundUnit(UnitQuantity):
    pass
