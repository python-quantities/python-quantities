# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

import operator

import numpy

from .config import USE_UNICODE
from .markup import format_units, format_units_unicode
from .registry import unit_registry


class Dimensionality(dict):

    """
    """

    @property
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
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self.copy()

    def __iadd__(self, other):
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self

    def __sub__(self, other):
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not subtract units of %s and %s'\
                %(str(self), str(other))
            )
        return self.copy()

    def __isub__(self, other):
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self

    def __mul__(self, other):
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
        for unit, power in other.iteritems():
            try:
                self[unit] += power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = power
        return self

    def __truediv__(self, other):
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
        return self.__truediv__(other)

    def __itruediv__(self, other):
        for unit, power in other.iteritems():
            try:
                self[unit] -= power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = -power
        return self

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        new = Dimensionality(self)
        for i in new:
            new[i] *= other
        return new

    def __ipow__(self, other):
        assert isinstance(other, (int, float))
        for i in self:
            self[i] *= other
        return self

    def __repr__(self):
        return self.string

    def __str__(self):
        if USE_UNICODE:
            return self.unicode
        else:
            return self.string

    def __eq__(self, other):
        return hash(self) == hash(other)

    def copy(self):
        return Dimensionality(self)
