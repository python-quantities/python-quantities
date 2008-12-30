import numpy


class BaseDimensionality(object):

    """
    """

    def _format_units(self, udict):
        num = []
        den = []
        for u, d in udict.iteritems():
            u = u.units
            if d>0:
                if d != 1: u = u + ('**%s'%d).rstrip('.0')
                num.append(u)
            elif d<0:
                d = -d
                if d != 1: u = u + ('**%s'%d).rstrip('.0')
                den.append(u)
        res = ' * '.join(num)
        if len(den):
            if not res: res = '1'
            res = res + ' / ' + ' '.join(den)
        if not res: res = '(dimensionless)'
        return res


    def __add__(self, other):
        assert self == other
        return MutableDimensionality(self)

    __sub__ = __add__

    def __mul__(self, other):
        new = MutableDimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] += power
            except KeyError:
                new[unit] = power
        return new

    def __div__(self, other):
        new = MutableDimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] -= power
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

#        del self.__dict__['__init__']

#    del __init__

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


class HasDimensionality(numpy.ndarray):

    def __new__(cls, magnitude, dtype='d', dimensionality={}, mutable=True):
        if not isinstance(magnitude, numpy.ndarray):
            magnitude = numpy.array(magnitude, dtype=dtype)
            if not magnitude.flags.contiguous:
                magnitude = magnitude.copy()

        ret = numpy.ndarray.__new__(
            cls,
            magnitude.shape,
            magnitude.dtype,
            buffer=magnitude
        )
        ret.flags.writeable = mutable
        return ret

    def __init__(self, data, dtype='d', dimensionality={}, mutable=True):
        if mutable:
            self._dimensionality = MutableDimensionality(dimensionality)
        else:
            self._dimensionality = ImmutableDimensionality(dimensionality)

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def magnitude(self):
        return self.view(type=numpy.ndarray)

    def __cmp__(self, other):
        raise

    def __add__(self, other):
        if self.dimensionality:
            assert isinstance(other, HasDimensionality)
        dims = self.dimensionality + other.dimensionality
        magnitude = self.magnitude + other.magnitude
        return Quantity(magnitude, magnitude.dtype, dims)

    def __sub__(self, other):
        if self.dimensionality:
            assert isinstance(other, HasDimensionality)
        dims = self.dimensionality - other.dimensionality
        magnitude = self.magnitude - other.magnitude
        return Quantity(magnitude, magnitude.dtype, dims)

    def __mul__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        try:
            dims = self.dimensionality * other.dimensionality
            magnitude = self.magnitude * other.magnitude
        except:
            dims = self.dimensionality
            magnitude = self.magnitude * other
        return Quantity(magnitude, magnitude.dtype, dims)

    def __div__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        try:
            dims = self.dimensionality / other.dimensionality
            magnitude = self.magnitude / other.magnitude
        except:
            dims = self.dimensionality
            magnitude = self.magnitude / other
        return Quantity(magnitude, magnitude.dtype, dims)

    def __pow__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        dims = self.dimensionality**other
        magnitude = self.magnitude**other
        return Quantity(magnitude, magnitude.dtype, dims)


class Quantity(HasDimensionality):

    def __init__(self, magnitude, dtype='d', units={}, mutable=True):
        if isinstance(units, HasDimensionality):
            units = units.dimensionality
        assert isinstance(units, (BaseDimensionality, dict))
        HasDimensionality.__init__(self, magnitude, dtype, units, mutable)

    def __repr__(self):
        return '%s*%s'%(numpy.ndarray.__str__(self), self.units)

    __str__ = __repr__

    @property
    def units(self):
        return str(self.dimensionality)


class UnitQuantity(Quantity):

    def __new__(cls, name, *args, **kwargs):
        return Quantity.__new__(
            cls,
            1.0,
            dtype='d',
            dimensionality={},
            mutable=False
        )

    def __init__(self, name, *args, **kwargs):
        self._name = name
        Quantity.__init__(self, 1.0, 'd', {self:1}, mutable=False)

    @property
    def reference_quantity(self):
        return self

    @property
    def units(self):
        try:
            return self._name
        except:
            return ''

    def __repr__(self):
        return self.units

    __str__ = __repr__

#    def __add__(self, other):
#        assert isinstance(other, HasDimensionality)
#        dims = HasDimensionality.__add__(self, other)
#        magnitude = self.magnitude + other.magnitude
#        return Quantity(dims)
#
#    __sub__ = __add__
#
#    def __mul__(self, other):
#        assert isinstance(other, (numpy.ndarray, int, float))
#        try:
#            dims = HasDimensionality.__mul__(self, other)
#            magnitude = self.magnitude * other.magnitude
#        except:
#            dims = self.dimensionality
#            magnitude = self.magnitude * other
#        return Quantity(magnitude, dims)
#
#    def __div__(self, other):
#        assert isinstance(other, (HasDimensionality, int, float))
#        try:
#            dims = HasDimensionality.__div__(self, other)
#            magnitude = self.magnitude / other.magnitude
#        except:
#            dims = self.dimensionality
#            magnitude = self.magnitude / other
#        return Quantity(magnitude, dims)
#
#    def __pow__(self, other):
#        assert isinstance(other, (int, float))
#        dims = HasDimensionality.__pow__(self, other)
#        return Quantity(self.magnitude**other, dims)


class ReferenceUnit(UnitQuantity):
    pass


class CompoundUnit(UnitQuantity):

    def __init__(self, name, reference_quantity):
        UnitQuantity.__init__(self, name)
        self._reference_quantity = reference_quantity

    @property
    def reference_quantity(self):
        return self._reference_quantity


m = ReferenceUnit('m')
kg = ReferenceUnit('kg')
s = ReferenceUnit('s')
J = CompoundUnit('J', kg*m**2/s**2)

energy = J*J

print energy, J, m
