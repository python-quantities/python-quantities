"""
"""

from quantities.quantity import Quantity
from quantities.parser import unit_registry
from quantities.dimensionality import ImmutableDimensionality


class UnitQuantity(Quantity):

    _primary_order = 99
    _secondary_order = 0

    def __new__(cls, name, reference_quantity=None):
        return Quantity.__new__(
            cls,
            1.0,
            units=None,
            dtype='d',
            mutable=False
        )

    def __init__(self, name, reference_quantity=None):
        self._name = name
        Quantity.__init__(self, 1.0, None, 'd', mutable=False)

        if reference_quantity is None:
            reference_quantity = self
        self._reference_quantity = reference_quantity

        self._format_order = (self._primary_order, self._secondary_order)
        self.__class__._secondary_order += 1

        unit_registry[name] = self

    @property
    def format_order(self):
        return self._format_order

    @property
    def name(self):
        return self._name

    @property
    def reference_quantity(self):
        return self._reference_quantity

    @property
    def units(self):
        return '(%s)'%self._name

unit_registry['UnitQuantity'] = UnitQuantity


class UnitMass(UnitQuantity):

    _primary_order = 0


class UnitLength(UnitQuantity):

    _primary_order = 1


class UnitTime(UnitQuantity):

    _primary_order = 2


class UnitCurrent(UnitQuantity):

    _primary_order = 3


class UnitLuminousIntensity(UnitQuantity):

    _primary_order = 4


class UnitSubstance(UnitQuantity):

    _primary_order = 5


class UnitTemperature(UnitQuantity):

    _primary_order = 6


class UnitInformation(UnitQuantity):

    _primary_order = 7


class UnitAngle(UnitQuantity):

    _primary_order = 8


class UnitCurrency(UnitQuantity):

    _primary_order = 9


class Dimensionless(UnitQuantity):

    def __init__(self, name, reference_quantity=None):
        self._name = name
        Quantity.__init__(self, 1.0, {}, 'd', mutable=False)

        if reference_quantity is None:
            reference_quantity = self
        self._reference_quantity = reference_quantity

        self._format_order = (self._primary_order, self._secondary_order)
        self.__class__._secondary_order += 1

        unit_registry[name] = self
