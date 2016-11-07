"""
"""
from __future__ import absolute_import

import sys
import weakref

import numpy

from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc


__all__ = [
    'CompoundUnit', 'Dimensionless', 'UnitConstant', 'UnitCurrency',
    'UnitCurrent', 'UnitInformation', 'UnitLength', 'UnitLuminousIntensity',
    'UnitMass', 'UnitMass', 'UnitQuantity', 'UnitSubstance', 'UnitTemperature',
    'UnitTime', 'set_default_units'
]


class UnitQuantity(Quantity):

    _primary_order = 90
    _secondary_order = 0
    _reference_quantity = None

    __array_priority__ = 20

    def __new__(
        cls, name, definition=None, symbol=None, u_symbol=None,
        aliases=[], doc=None
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
        if doc is not None:
            ret.__doc__ = doc

        if definition is not None:
            if not isinstance(definition, Quantity):
                definition *= dimensionless
            ret._definition = definition
            ret._conv_ref = definition._reference
        else:
            ret._definition = None
            ret._conv_ref = None

        ret._aliases = aliases

        ret._format_order = (ret._primary_order, ret._secondary_order)
        ret.__class__._secondary_order += 1

        return ret

    def __init__(
        self, name, definition=None, symbol=None, u_symbol=None,
        aliases=[], doc=None
    ):
        unit_registry[name] = self
        if symbol:
            unit_registry[symbol] = self
        for alias in aliases:
            unit_registry[alias] = self

    def __array_finalize__(self, obj):
        pass

    def __hash__(self):
        return hash((type(self), self._name))

    @property
    def _reference(self):
        if self._conv_ref is None:
            return self
        else:
            return self._conv_ref

    @property
    def _dimensionality(self):
        return Dimensionality({self:1})

    @property
    def format_order(self):
        return self._format_order

    @property
    def name(self):
        return self._name

    @property
    def definition(self):
        if self._definition is None:
            return self
        else:
            return self._definition

    @property
    def simplified(self):
        return self._reference.simplified

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
    @units.setter
    def units(self, units):
        raise AttributeError('can not modify protected units')

    def __repr__(self):
        ref = self._definition
        if ref:
            ref = ', %s * %s'%(str(ref.magnitude), ref.dimensionality.string)
        else:
            ref = ''
        symbol = self._symbol
        symbol = ', %s'%(repr(symbol)) if symbol else ''
        if markup.config.use_unicode:
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
            if markup.config.use_unicode:
                s = '1 %s (%s)'%(self.u_symbol, self.name)
            else:
                s = '1 %s (%s)'%(self.symbol, self.name)
        else:
            s = '1 %s'%self.name

        return s

    @with_doc(Quantity.__add__, use_header=False)
    def __add__(self, other):
        return self.view(Quantity).__add__(other)

    @with_doc(Quantity.__radd__, use_header=False)
    def __radd__(self, other):
        try:
            return self.rescale(other.units).__radd__(other)
        except AttributeError:
            return self.view(Quantity).__radd__(other)

    @with_doc(Quantity.__sub__, use_header=False)
    def __sub__(self, other):
        return self.view(Quantity).__sub__(other)

    @with_doc(Quantity.__rsub__, use_header=False)
    def __rsub__(self, other):
        try:
            return self.rescale(other.units).__rsub__(other)
        except AttributeError:
            return self.view(Quantity).__rsub__(other)

    @with_doc(Quantity.__mod__, use_header=False)
    def __mod__(self, other):
        return self.view(Quantity).__mod__(other)

    @with_doc(Quantity.__rsub__, use_header=False)
    def __rmod__(self, other):
        try:
            return self.rescale(other.units).__rmod__(other)
        except AttributeError:
            return self.view(Quantity).__rmod__(other)

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

    if sys.version_info[0] < 3:
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

    if sys.version_info[0] < 3:
        @with_doc(Quantity.__idiv__, use_header=False)
        def __idiv__(self, other):
            raise TypeError('can not modify protected units')

    @with_doc(Quantity.__ipow__, use_header=False)
    def __ipow__(self, other):
        raise TypeError('can not modify protected units')

    def __getstate__(self):
        """
        Return the internal state of the quantity, for pickling
        purposes.

        """
        state = (1, self._format_order)
        return state

    def __setstate__(self, state):
        ver, fo = state
        self._format_order = fo

    def __reduce__(self):
        """
        Return a tuple for pickling a UnitQuantity.
        """
        return (
            type(self),
            (
                self._name,
                self._definition,
                self._symbol,
                self._u_symbol,
                self._aliases,
                self.__doc__
            ),
            self.__getstate__()
        )

    def copy(self):
        return (
            type(self)(
                self._name,
                self._definition,
                self._symbol,
                self._u_symbol,
                self._aliases,
                self.__doc__
                )
            )

unit_registry['UnitQuantity'] = UnitQuantity


class IrreducibleUnit(UnitQuantity):

    _default_unit = None

    def __init__(
        self, name, definition=None, symbol=None, u_symbol=None,
        aliases=[], doc=None
    ):
        super(IrreducibleUnit, self).__init__(
            name, definition, symbol, u_symbol, aliases, doc
        )
        cls = type(self)
        if cls._default_unit is None:
            cls._default_unit = self

    @property
    def simplified(self):
        return self.view(Quantity).rescale(self.get_default_unit())

    @classmethod
    def get_default_unit(cls):
        return cls._default_unit
    @classmethod
    def set_default_unit(cls, unit):
        if unit is None:
            return
        if isinstance(unit, str):
            unit = unit_registry[unit]
        try:
            # check that conversions are possible:
            get_conversion_factor(cls._default_unit, unit)
        except ValueError:
            raise TypeError('default unit must be of same type')
        cls._default_unit = unit


class UnitMass(IrreducibleUnit):

    _primary_order = 1


class UnitLength(IrreducibleUnit):

    _primary_order = 2


class UnitTime(IrreducibleUnit):

    _primary_order = 3


class UnitCurrent(IrreducibleUnit):

    _primary_order = 4


class UnitLuminousIntensity(IrreducibleUnit):

    _primary_order = 5


class UnitSubstance(IrreducibleUnit):

    _primary_order = 6


class UnitTemperature(IrreducibleUnit):

    _primary_order = 7


class UnitInformation(IrreducibleUnit):

    _primary_order = 8


class UnitCurrency(IrreducibleUnit):

    _primary_order = 9


class CompoundUnit(UnitQuantity):

    _primary_order = 99

    def __new__(cls, name):
        return UnitQuantity.__new__(cls, name, unit_registry[name])

    def __init__(self, name):
        # do not register
        return

    @with_doc(UnitQuantity.__add__, use_header=False)
    def __repr__(self):
        return '1 %s'%self.name

    @property
    def name(self):
        if markup.config.use_unicode:
            return '(%s)'%(markup.superscript(self._name))
        else:
            return '(%s)'%self._name

    def __reduce__(self):
        """
        Return a tuple for pickling a UnitQuantity.
        """
        return (
            type(self),
            (self._name, ),
            self.__getstate__()
            )

    def copy(self):
        return type(self)(self._name)

unit_registry['CompoundUnit'] = CompoundUnit


class Dimensionless(UnitQuantity):

    _primary_order = 100

    def __init__(self, name, definition=None):
        self._name = name

        if definition is None:
            definition = self
        self._definition = definition

        self._format_order = (self._primary_order, self._secondary_order)
        self.__class__._secondary_order += 1

        unit_registry[name] = self

    def __reduce__(self):
        """
        Return a tuple for pickling a UnitQuantity.
        """
        return (
            type(self),
            (
                self._name,
            ),
            self.__getstate__()
        )

    @property
    def _dimensionality(self):
        return Dimensionality()

dimensionless = Dimensionless('dimensionless')


class UnitConstant(UnitQuantity):

    _primary_order = 0

    def __init__(
        self, name, definition=None, symbol=None, u_symbol=None,
        aliases=[], doc=None
    ):
        # we dont want to register constants in the unit registry
        return


def set_default_units(
    system=None, currency=None, current=None, information=None, length=None,
    luminous_intensity=None, mass=None, substance=None, temperature=None,
    time=None
):
    """
    Set the default units in which simplified quantities will be
    expressed.

    system sets the unit system, and can be "SI" or "cgs". All other
    keyword arguments will accept either a string or a unit quantity.
    An error will be raised if it is not possible to convert between
    old and new defaults, so it is not possible to set "kg" as the
    default unit for time.

    If both system and individual defaults are given, the system
    defaults will be applied first, followed by the individual ones.
    """
    if system is not None:
        system = system.lower()
        try:
            assert system in ('si', 'cgs')
        except AssertionError:
            raise ValueError('system must be "SI" or "cgs", got "%s"' % system)
        if system == 'si':
            UnitCurrent.set_default_unit('A')
            UnitLength.set_default_unit('m')
            UnitMass.set_default_unit('kg')
        elif system == 'cgs':
            UnitLength.set_default_unit('cm')
            UnitMass.set_default_unit('g')
        UnitLuminousIntensity.set_default_unit('cd')
        UnitSubstance.set_default_unit('mol')
        UnitTemperature.set_default_unit('degK')
        UnitTime.set_default_unit('s')

    UnitCurrency.set_default_unit(currency)
    UnitCurrent.set_default_unit(current)
    UnitInformation.set_default_unit(information)
    UnitLength.set_default_unit(length)
    UnitLuminousIntensity.set_default_unit(luminous_intensity)
    UnitMass.set_default_unit(mass)
    UnitSubstance.set_default_unit(substance)
    UnitTemperature.set_default_unit(temperature)
    UnitTime.set_default_unit(time)
