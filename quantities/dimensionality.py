# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

import operator

import numpy

from .markup import USE_UNICODE, format_units, format_units_unicode
from .registry import unit_registry


class Dimensionality(object):

    """
    """

    def __init__(self, dict=None):
        self.__data = {}
        if dict is not None:
            self.__data.update(dict)

    @property
    def simplified(self):
        if len(self):
            rq = (1*unit_registry['dimensionless']).copy()
            for u, d in self.iteritems():
                rq = rq * u.simplified**d
            return rq.dimensionality
        else:
            return self

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

    def __sub__(self, other):
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not subtract units of %s and %s'\
                %(str(self), str(other))
            )
        return self.copy()

    def __mul__(self, other):
        new = dict(self)
        for unit, power in other.iteritems():
            try:
                new[unit] += power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = power
        return Dimensionality(new)

    def __truediv__(self, other):
        new = dict(self)
        for unit, power in other.iteritems():
            try:
                new[unit] -= power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = -power
        return Dimensionality(new)

    def __div__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        new = dict(self)
        for i in new:
            new[i] *= other
        return Dimensionality(new)

    def __repr__(self):
        if USE_UNICODE:
            return self.unicode()
        else:
            return self.string()

    def __cmp__(self, dict):
        if isinstance(dict, Dimensionality):
            return cmp(self.__data, dict.__data)
        else:
            return cmp(self.__data, dict)

    def __len__(self):
        return len(self.__data)

    def __contains__(self, key):
        return key in self.__data

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __getitem__(self, key):
        return self.__data[key]

    def __iter__(self):
        return self.__data.__iter__()

    def copy(self):
        return Dimensionality(self.__data.copy())

    def keys(self):
        return self.__data.keys()

    def items(self):
        return self.__data.items()

    def iteritems(self):
        return self.__data.iteritems()

    def iterkeys(self):
        return self.__data.iterkeys()

    def itervalues(self):
        return self.__data.itervalues()

    def values(self):
        return self.__data.values()

    def has_key(self, key):
        return self.__data.has_key(key)

    def get(self, key, failobj=None):
        if not self.has_key(key):
            return failobj
        return self[key]

    def string(self):
        return format_units(self)

    def unicode(self):
        return format_units_unicode(self)
