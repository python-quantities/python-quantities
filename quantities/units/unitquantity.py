"""
"""

import numpy

from quantities.dimensionality import ImmutableDimensionality
from quantities.quantity import Quantity
from quantities.registry import unit_registry


class UnitQuantity(Quantity):

    _primary_order = 99
    _secondary_order = 0

    def __new__(
        cls, name, reference_quantity=None, symbol=None, aliases=[], note=None
    ):
        data = numpy.array(1, dtype='d')
        ret = numpy.ndarray.__new__(
            cls,
            data.shape,
            data.dtype,
            buffer=data
        )
        ret.flags.writeable = False
        return ret

    def __init__(
        self, name, reference_quantity=None, symbol=None, aliases=[], note=None
    ):
        self._name = name
        self._symbol = symbol
        self._note = note
        self._dimensionality = ImmutableDimensionality({self:1})

        if reference_quantity is None:
            self._reference_quantity = self
        else:
            self._reference_quantity = reference_quantity.simplified

        self._format_order = (self._primary_order, self._secondary_order)
        self.__class__._secondary_order += 1

        unit_registry[name] = self
        if symbol:
            unit_registry[symbol] = self
        for alias in aliases:
            unit_registry[alias] = self

    def __repr__(self):
        if self._symbol:
            s = '1 %s (%s)'%(self.symbol, self.name)
        else:
            s = '1 %s'%self.name

        if self.note:
            return s+'\nnote: %s'%self.note
        return s

    @property
    def format_order(self):
        return self._format_order

    @property
    def name(self):
        return self._name

    @property
    def note(self):
        return self._note

    @property
    def reference_quantity(self):
        return self._reference_quantity

    @property
    def symbol(self):
        if self._symbol:
            return self._symbol
        else:
            return self.name

    @property
    def units(self):
        return self

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
        self._dimensionality = ImmutableDimensionality({})

        if reference_quantity is None:
            reference_quantity = self
        self._reference_quantity = reference_quantity

        self._format_order = (self._primary_order, self._secondary_order)
        self.__class__._secondary_order += 1

        unit_registry[name] = self

dimensionless = Dimensionless('dimensionless')
