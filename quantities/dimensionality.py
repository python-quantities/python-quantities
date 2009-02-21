# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

import operator

import numpy

from .config import USE_UNICODE
from .markup import format_units, format_units_unicode
from .registry import unit_registry
from .utilities import memoize

def assert_isinstance(obj, types):
    try:
        assert isinstance(obj, types)
    except AssertionError:
        raise TypeError(
            "arg %r must be of type %r, got %r" % (obj, types, type(obj))
        )


class Dimensionality(dict):

    """
    """

    @property
    def ndims(self):
        return sum(abs(i) for i in self.simplified.itervalues())

    @property
    @memoize
    def simplified(self):
        if len(self):
            rq = 1*unit_registry['dimensionless']
            for u, d in self.iteritems():
                rq = rq * u.simplified**d
            return rq.dimensionality
        else:
            return self

    @property
    def string(self):
        return format_units(self)

    @property
    def unicode(self):
        return format_units_unicode(self)

    def __hash__(self):
        res = hash(unit_registry['dimensionless'])
        for key in sorted(self.keys(), key=operator.attrgetter('format_order')):
            val = self[key]
            res ^= hash((key, val))
        return res

    def __add__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self.copy()

    __radd__ = __add__

    def __iadd__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self

    def __sub__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not subtract units of %s and %s'\
                %(str(self), str(other))
            )
        return self.copy()

    __rsub__ = __sub__

    def __isub__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self

    def __mul__(self, other):
        assert_isinstance(other, Dimensionality)
        new = Dimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] += power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = power
        return new

    def __imul__(self, other):
        assert_isinstance(other, Dimensionality)
        for unit, power in other.iteritems():
            try:
                self[unit] += power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = power
        return self

    def __truediv__(self, other):
        assert_isinstance(other, Dimensionality)
        new = Dimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] -= power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = -power
        return new

    def __div__(self, other):
        assert_isinstance(other, Dimensionality)
        return self.__truediv__(other)

    def __itruediv__(self, other):
        assert_isinstance(other, Dimensionality)
        for unit, power in other.iteritems():
            try:
                self[unit] -= power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = -power
        return self

    def __idiv__(self, other):
        assert_isinstance(other, Dimensionality)
        return self.__itruediv__(other)

    def __pow__(self, other):
        try:
            assert numpy.isscalar(other)
        except AssertionError:
            raise TypeError('exponent must be a scalar, got %r' % other)
        if other == 0:
            return Dimensionality()
        new = Dimensionality(self)
        for i in new:
            new[i] *= other
        return new

    def __ipow__(self, other):
        try:
            assert numpy.isscalar(other)
        except AssertionError:
            raise TypeError('exponent must be a scalar, got %r' % other)
        if other == 0:
            self.clear()
            return self
        for i in self:
            self[i] *= other
        return self

    def __repr__(self):
        return 'Dimensionality({%s})' \
            % ', '.join(['%s: %s'% (u.name, e) for u, e in self.iteritems()])

    def __str__(self):
        if USE_UNICODE:
            return self.unicode
        else:
            return self.string

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return hash(self) != hash(other)

    __neq__ = __ne__

    def __gt__(self, other):
        return self.ndims > other.ndims

    def __ge__(self, other):
        return self.ndims >= other.ndims

    def __lt__(self, other):
        return self.ndims < other.ndims

    def __le__(self, other):
        return self.ndims <= other.ndims

    def copy(self):
        return Dimensionality(self)


p_dict = {}

def _d_multiply(q1, q2):
    try:
        return q1._dimensionality * q2._dimensionality
    except AttributeError:
        try:
            return q1.dimensionality
        except:
            return q2.dimensionality
p_dict[numpy.multiply] = _d_multiply

def _d_divide(q1, q2):
    try:
        return q1._dimensionality / q2._dimensionality
    except AttributeError:
        try:
            return q1.dimensionality
        except:
            return q2.dimensionality**-1
p_dict[numpy.divide] = _d_divide
p_dict[numpy.true_divide] = _d_divide

def _d_add_sub(q1, q2):
    try:
        return q1._dimensionality + q2._dimensionality
    except AttributeError:
        if hasattr(q1, 'dimensionality'):
            return q1.dimensionality
        elif hasattr(q2, 'dimensionality'):
            return q2.dimensionality
p_dict[numpy.add] = _d_add_sub
p_dict[numpy.subtract] = _d_add_sub

def _d_power(q1, q2):
    if getattr(q2, 'dimensionality', None):
        raise ValueError("exponent must be dimensionless")
    try:
        q2 = numpy.array(q2)
        p = q2.min()
        if p != q2.max():
            raise ValueError('Quantities must be raised to a uniform power')
        return q1._dimensionality**p
    except AttributeError:
        return Dimensionality()
p_dict[numpy.power] = _d_power

def _d_square(q1):
    return q1._dimensionality**2
p_dict[numpy.square] = _d_square

def _d_reciprocal(q1):
    return q1._dimensionality**-1
p_dict[numpy.reciprocal] = _d_reciprocal

def _d_copy(q1):
    return q1.dimensionality
p_dict[numpy.conjugate] = _d_copy
p_dict[numpy.floor] = _d_copy
p_dict[numpy.ceil] = _d_copy
p_dict[numpy.rint] = _d_copy

def _d_sqrt(q1):
    return q1._dimensionality**0.5
p_dict[numpy.sqrt] = _d_sqrt
