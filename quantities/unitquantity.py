"""
"""
from __future__ import absolute_import

import numpy

from .config import USE_UNICODE
from .dimensionality import Dimensionality
from .markup import superscript
from .quantity import Quantity
from .registry import unit_registry
from .utilities import with_doc


__all__ = [
    'CompoundUnit', 'Dimensionless', 'UnitAngle', 'UnitConstant',
    'UnitCurrency', 'UnitCurrent', 'UnitInformation', 'UnitLength',
    'UnitLuminousIntensity', 'UnitMass', 'UnitMass', 'UnitQuantity',
    'UnitSubstance', 'UnitTemperature', 'UnitTime'
]


class UnitQuantity(Quantity):

    _primary_order = 90
    _secondary_order = 0
    _reference_quantity = None

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

        # handle dimensionality in the property
        ret._dimensionality = None

        ret._reference_quantity = reference_quantity

        if reference_quantity is not None:
            if not isinstance(reference_quantity, Quantity):
                reference_quantity *= dimensionless
            ret._simplified = reference_quantity.simplified
        else:
            ret._simplified = None

        ret._format_order = (ret._primary_order, ret._secondary_order)
        ret.__class__._secondary_order += 1

        return ret

    def __init__(
        self, name, reference_quantity=None, symbol=None, u_symbol=None,
        aliases=[], note=None
    ):
        unit_registry[name] = self
        if symbol:
            unit_registry[symbol] = self
        for alias in aliases:
            unit_registry[alias] = self

    @with_doc(Quantity.__repr__, use_header=False)
    def __repr__(self):
        ref = self._reference_quantity
        if ref:
            ref = ', %s*%s'%(str(ref.magnitude), repr(ref.dimensionality))
        else:
            ref = ''
        symbol = self._symbol
        symbol = ', %s'%(repr(symbol)) if symbol else ''
        if USE_UNICODE:
            u_symbol = self._u_symbol
            u_symbol = ', %s'%(repr(u_symbol)) if u_symbol else ''
        else:
            u_symbol = ''
        return '%s(%s%s%s%s)'%(
            self.__class__.__name__, repr(self.name), ref, symbol, u_symbol
        )

    @with_doc(Quantity.__str__, use_header=False)
    def __str__(self):
        if self.u_symbol != self.name:
            if USE_UNICODE:
                s = '1 %s (%s)'%(self.u_symbol, self.name)
            else:
                s = '1 %s (%s)'%(self.symbol, self.name)
        else:
            s = '1 %s'%self.name

#        if self.note:
#            return s+'\nnote: %s'%self.note
        return s

    @with_doc(Quantity.__add__, use_header=False)
    def __add__(self, other):
        return self.view(Quantity).__add__(other)

    @with_doc(Quantity.__radd__, use_header=False)
    def __radd__(self, other):
        return self.view(Quantity).__radd__(other)

    @with_doc(Quantity.__sub__, use_header=False)
    def __sub__(self, other):
        return self.view(Quantity).__sub__(other)

    @with_doc(Quantity.__rsub__, use_header=False)
    def __rsub__(self, other):
        return self.view(Quantity).__rsub__(other)

    @with_doc(Quantity.__mul__, use_header=False)
    def __mul__(self, other):
        return self.view(Quantity).__mul__(other)

    @with_doc(Quantity.__rmul__, use_header=False)
    def __rmul__(self, other):
        return self.view(Quantity).__rmul__(other)

    @with_doc(Quantity.__truediv__, use_header=False)
    def __truediv__(self, other):
        return self.view(Quantity).__truediv__(other)

    @with_doc(Quantity.__rtruediv__, use_header=False)
    def __rtruediv__(self, other):
        return self.view(Quantity).__rtruediv__(other)

    @with_doc(Quantity.__div__, use_header=False)
    def __div__(self, other):
        return self.view(Quantity).__div__(other)

    @with_doc(Quantity.__rdiv__, use_header=False)
    def __rdiv__(self, other):
        return self.view(Quantity).__rdiv__(other)

    @with_doc(Quantity.__pow__, use_header=False)
    def __pow__(self, other):
        return self.view(Quantity).__pow__(other)

    @with_doc(Quantity.__rpow__, use_header=False)
    def __rpow__(self, other):
        return self.view(Quantity).__rpow__(other)

    @with_doc(Quantity.__iadd__, use_header=False)
    def __iadd__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__isub__, use_header=False)
    def __isub__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__imul__, use_header=False)
    def __imul__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__itruediv__, use_header=False)
    def __itruediv__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__idiv__, use_header=False)
    def __idiv__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__ipow__, use_header=False)
    def __ipow__(self, other):
        raise TypeError('can not modify protected units')

    @property
    def dimensionality(self):
        return Dimensionality({self:1})

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
    def simplified(self):
        if self._simplified is not None:
            return self._simplified
#            return self.reference_quantity.simplified
            # if alternat unit system:
            # return self.reference_quantity.simplified.simplified
        else:
            return self
            # if alternate units:
            # return self.rescale(alt)

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

    def _get_units(self):
        return self
    def _set_units(self, units):
        raise AttributeError('can not modify protected units')
    units = property(_get_units, _set_units)

    @classmethod
    def set_reference_unit(cls, unit):
        if cls.__name__ in ('UnitConstant', 'UnitQuantity', 'Dimensionless'):
            raise ValueError(
                'can not set alternate reference unit for type %'% cls.__name__
            )
        if isinstance(unit, str):
            unit = unit_registry[unit]
        try:
            assert type(unit) == cls or unit is None
        except:
            raise TypeError('unit must be of same type or "None"')

        cls._alt_reference_unit = unit

    @property
    def alt_reference_unit(self):
        return self.__class__._alt_reference_unit


unit_registry['UnitQuantity'] = UnitQuantity


class UnitConstant(UnitQuantity):

    _primary_order = 0

    def __init__(
        self, name, reference_quantity=None, symbol=None, u_symbol=None,
        aliases=[], note=None
    ):
        # we dont want to register constants in the unit registry
        return


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


class CompoundUnit(UnitQuantity):

    _primary_order = 99

    def __new__(cls, name):
        return UnitQuantity.__new__(cls, name, unit_registry[name])

    @with_doc(UnitQuantity.__add__, use_header=False)
    def __repr__(self):
        return '1 %s'%self.name

    @property
    def name(self):
        if USE_UNICODE:
            return '(%s)'%(superscript(self._name))
        else:
            return '(%s)'%self._name

unit_registry['CompoundUnit'] = CompoundUnit


class Dimensionless(UnitQuantity):

    _primary_order = 100

    def __init__(self, name, reference_quantity=None):
        self._name = name
        self._dimensionality = None

        if reference_quantity is None:
            reference_quantity = self
        self._reference_quantity = reference_quantity

        self._format_order = (self._primary_order, self._secondary_order)
        self.__class__._secondary_order += 1

        unit_registry[name] = self

    @property
    def dimensionality(self):
        return Dimensionality()

dimensionless = Dimensionless('dimensionless')
