"""
"""

import numpy

from quantities.dimensionality import ImmutableDimensionality
from quantities.quantity import Quantity
from quantities.registry import unit_registry

__all__ = [
    'Dimensionless', 'UnitAngle', 'UnitCurrency', 'UnitCurrent',
    'UnitInformation', 'UnitLength', 'UnitLuminousIntensity', 'UnitMass',
    'UnitMass', 'UnitQuantity', 'UnitSubstance', 'UnitTemperature', 'UnitTime'
]


def quantity(f):

    def wrapped(*args, **kwargs):
        ret = f(*args, **kwargs)
        if isinstance(ret, UnitQuantity):
            return ret.view(Quantity).copy()
        return ret

    return wrapped


class UnitQuantity(Quantity):

    _primary_order = 99
    _secondary_order = 0

    __array_priority__ = 20

    def __new__(
        cls, name, reference_quantity=None, symbol=None, u_symbol=None,
        aliases=[], note=None
    ):
        try:
            assert isinstance(name, str)
        except AssertionError:
            raise TypeError('name must be a string, got %s (not unicode)'%name)
        try:
            assert symbol is None or isinstance(symbol, str)
        except AssertionError:
            raise TypeError(
                'symbol must be a string, '
                'got %s (u_symbol can be unicode)'%symbol
            )

        ret = numpy.array(1, dtype='d').view(cls)
        ret.flags.writeable = False

        ret._name = name
        ret._symbol = symbol
        ret._u_symbol = u_symbol
        ret._note = note
        ret._dimensionality = ImmutableDimensionality({ret:1})

        try:
            reference_quantity = reference_quantity.simplified
        except AttributeError:
            pass
        ret._reference_quantity = reference_quantity

        ret._format_order = (ret._primary_order, ret._secondary_order)
        ret.__class__._secondary_order += 1

        unit_registry[name] = ret
        if symbol:
            unit_registry[symbol] = ret
        for alias in aliases:
            unit_registry[alias] = ret

        return ret

    def __repr__(self):
        if self._symbol:
            s = '1 %s (%s)'%(self.symbol, self.name)
        else:
            s = '1 %s'%self.name

        if self.note:
            return s+'\nnote: %s'%self.note
        return s

    @quantity
    def __mul__(self, other):
        return super(UnitQuantity, self).__mul__(other)

    @quantity
    def __rmul__(self, other):
        return super(UnitQuantity, self).__rmul__(other)

    @quantity
    def __truediv__(self, other):
        return super(UnitQuantity, self).__truediv__(other)

    @quantity
    def __rtruediv__(self, other):
        return super(UnitQuantity, self).__rtruediv__(other)

    @quantity
    def __div__(self, other):
        return super(UnitQuantity, self).__div__(other)
    @quantity
    def __rdiv__(self, other):
        return super(UnitQuantity, self).__rdiv__(other)

    def __imul__(self, other):
        raise TypeError('can not modify protected units')

    def __itruediv__(self, other):
        raise TypeError('can not modify protected units')

    def __idiv__(self, other):
        raise TypeError('can not modify protected units')


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
        if self._reference_quantity is not None:
            return self._reference_quantity
        else:
            return self

    @property
    def symbol(self):
        if self._symbol:
            return self._symbol
        else:
            return self.name

    @property
    def u_symbol(self):
        if self._u_symbol:
            return self._u_symbol
        else:
            return self.symbol

    @property
    def units(self):
        return self

unit_registry['UnitQuantity'] = UnitQuantity


class UnitMass(UnitQuantity):

    _primary_order = 1


class UnitLength(UnitQuantity):

    _primary_order = 2


class UnitTime(UnitQuantity):

    _primary_order = 3


class UnitCurrent(UnitQuantity):

    _primary_order = 4


class UnitLuminousIntensity(UnitQuantity):

    _primary_order = 5


class UnitSubstance(UnitQuantity):

    _primary_order = 6


class UnitTemperature(UnitQuantity):

    _primary_order = 7


class UnitInformation(UnitQuantity):

    _primary_order = 8


class UnitAngle(UnitQuantity):

    _primary_order = 9


class UnitCurrency(UnitQuantity):

    _primary_order = 10


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
