"""
"""

import copy

import numpy

from quantities.dimensionality import BaseDimensionality, \
    MutableDimensionality, ImmutableDimensionality
from quantities.parser import unit_registry


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

#    def __array_finalize__(self, obj):
#        self._dimensionality = copy.deepcopy(
#            getattr(obj, '_dimensionality', None)
#        )
#
#    def __deepcopy__(self, memo={}):
#        dimensionality = copy.deepcopy(self.dimensionality)
#        return self.__class__(
#            self.view(type=ndarray),
#            self.dtype,
#            dimensionality
#        )

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

    def __rmul__(self, other):
        # TODO: This needs to be properly implemented
        return self.__mul__(other)

    def __rdiv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        dims = self.dimensionality**other
        magnitude = self.magnitude**other
        return Quantity(magnitude, magnitude.dtype, dims)


class Quantity(HasDimensionality):

    __array_priority__ = 21

    def __init__(self, magnitude, dtype='d', units={}, mutable=True):
        if isinstance(units, str):
            units = unit_registry[units].dimensionality
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
