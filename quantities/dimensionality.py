"""
"""

import operator

import numpy

from quantities.registry import unit_registry

def format_units(udict):
    '''
    create a string representation of the units contained in a dimensionality
    '''
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
        u = key.name
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
        fmt = '(%s)' if len(den) > 1 else '%s'
        res = res + '/' + fmt%('*'.join(den))
    if not res: res = 'dimensionless'
    return '%s'%res


class BaseDimensionality(object):

    """
    """

    @property
    def simplified(self):
        if len(self):
            rq = 1
            for u, d in self.iteritems():
                rq = rq * u.reference_quantity**d
            return rq.dimensionality
        else:
            return self

    def __add__(self, other):
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return Dimensionality(self)

    def __sub__(self, other):
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not subtract units of %s and %s'\
                %(str(self), str(other))
            )
        return Dimensionality(self)

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

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        new = Dimensionality(self)
        for i in new:
            #multiply all the entries by the power
            new[i] *= other
        return new

class ImmutableDimensionality(BaseDimensionality):

    def __init__(self, dict=None):
        self.__data = {}
        if dict is not None:
            self.__data.update(dict)

    def __repr__(self):
        return format_units(self.__data)

    def __cmp__(self, dict):
        if isinstance(dict, ImmutableDimensionality):
            return cmp(self.__data, dict.__data)
        else:
            return cmp(self.__data, dict)

    def __len__(self):
        return len(self.__data)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __getitem__(self, key):
        return self.__data[key]

    def __hash__(self):
        res = hash(unit_registry['dimensionless'])
        for item in self.items():
            res ^= hash(item)
        return res

    def __iter__(self):
        return self.__data.__iter__()

    def copy(self):
        return ImmutableDimensionality(self.__data.copy())

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


class Dimensionality(BaseDimensionality, dict):

    def __repr__(self):
        return format_units(self)
