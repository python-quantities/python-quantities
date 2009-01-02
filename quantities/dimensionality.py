"""
"""

import operator

import numpy

from quantities.parser import unit_registry

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
        fmt = '(%s)' if len(den) > 1 else '%s'
        res = res + '/' + fmt%('*'.join(den))
    if not res: res = 'dimensionless'
    return '%s'%res


class BaseDimensionality(object):

    """
    """

    @property
    def udunits(self):
        """
        string representation of the unit group in the udunits format
        """
        return str(self).replace('**', '^')

    def __add__(self, other):
        try:
            # in order to allow adding different units (i.e. ft + m) need to
            # compare the two fully reduced units
            assert self == other
        except AssertionError:
            raise TypeError(
                'can not add quantities of with units of %s and %s'\
                %(str(self), str(other))
            )
        return MutableDimensionality(self)

    __sub__ = __add__

    def __mul__(self, other):
        #make a new dimensionality object for the result from the first object
        new = MutableDimensionality(self)
        for unit, power in other.iteritems():
            try:
                #add existing units together
                new[unit] += power
                #if the unit has a zero power, remove it
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                #if we get a keyerror, the unit does not exist in the first
                #dimensionality, so add it in
                new[unit] = power
        return new

    def __div__(self, other):
        new = MutableDimensionality(self)
        for unit, power in other.iteritems():
            try:
                #add the power to the entry for the unit
                new[unit] -= power
                #if the unit is raised to the zeroth power, remove it
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                #if we get an exception, then the unit did not exist before
                #so we have to add it in
                new[unit] = -power
        return new

    def __pow__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        if isinstance(other, numpy.ndarray):
            try:
                #make sure that if an array is used to power a unit,
                #the array just repeats the same number
                assert other.min()==other.max()
                other = other.min()
            except AssertionError:
                raise ValueError('Quantities must be raised to a single power')

        new = MutableDimensionality(self)
        for i in new:
            #multiply all the entries by the power
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
        if self.__class__ is ImmutableDimensionality:
            return ImmutableDimensionality(self.__data.copy())
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
        return format_units(self)
