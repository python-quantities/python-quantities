"""
"""

import operator

import numpy


class IncompatibleUnits(Exception):

    def __init__(self, op, operand1, operand2):
        self._op = op
        self._op1 = operand1
        self._op2 = operand2
        return

    def __str__(self):
        str = "Cannot %s quanitites with units of '%s' and '%s'" % \
              (self._op, self._op1, self._op2)
        return str


class BaseDimensionality(object):

    """
    """

    def _format_units(self, udict):
        num = []
        den = []
        keys = [k for k, o in
            sorted(
                [(k, k.format_order) for k in udict],
                key=operator.itemgetter(1)
            )
        ]
        for key in keys:
            d = udict[key]
            u = key.units
            if d>0:
                if d != 1: u = u + ('**%s'%d).rstrip('.0')
                num.append(u)
            elif d<0:
                d = -d
                if d != 1: u = u + ('**%s'%d).rstrip('.0')
                den.append(u)
        res = '*'.join(num)
        if len(den):
            if not res: res = '1'
            res = res + '/' + '*'.join(den)
        if not res: res = 'dimensionless'
        return '(%s)'%res


    def __add__(self, other):
        assert self == other
        return MutableDimensionality(self)

    __sub__ = __add__

    def __mul__(self, other):
        new = MutableDimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] += power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = power
        return new

    def __div__(self, other):
        new = MutableDimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] -= power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = -power
        return new

    def __pow__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        if isinstance(other, numpy.ndarray):
            try:
                assert other.min()==other.max()
                other = other.min()
            except AssertionError:
                raise ValueError('Quantities must be raised to a single power')

        new = MutableDimensionality(self)
        for i in new:
            new[i] *= other
        return new


class ImmutableDimensionality(BaseDimensionality):

    def __init__(self, dict=None, **kwds):
        self.__data = {}
        if dict is not None:
            self.__data.update(dict)
        if len(kwds):
            self.__data.update(kwds)

    def __repr__(self):
        return self._format_units(self.__data)

    def __cmp__(self, dict):
        if isinstance(dict, tuct):
            return cmp(self.__data, dict.__data)
        else:
            return cmp(self.__data, dict)

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, key):
        return self.__data[key]

    def __hash__(self):
        items = self.items()
        res = hash(items[0])
        for item in items[1:]:
            res ^= hash(item)
        return res

    def copy(self):
        if self.__class__ is tuct:
            return tuct(self.__data.copy())
        import copy
        __data = self.__data
        try:
            self.__data = {}
            c = copy.copy(self)
        finally:
            self.__data = __data
        c.update(self)
        return c

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

    def __contains__(self, key):
        return key in self.__data

    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d


class MutableDimensionality(BaseDimensionality, dict):

    def __repr__(self):
        return self._format_units(self)
