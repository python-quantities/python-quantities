"""
"""

import copy

import numpy

from quantities.dimensionality import BaseDimensionality, \
    MutableDimensionality, ImmutableDimensionality
from quantities.parser import unit_registry


class Quantity(numpy.ndarray):

    # TODO: what is an appropriate value?
    __array_priority__ = 21

    def __new__(cls, magnitude, units={}, dtype='d', mutable=True):
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

    def __init__(self, data, units={}, dtype='d', mutable=True):
        if isinstance(units, str):
            units = unit_registry[units].dimensionality
        if isinstance(units, Quantity):
            units = units.dimensionality
        assert isinstance(units, (BaseDimensionality, dict))

        if mutable:
            self._dimensionality = MutableDimensionality(units)
        else:
            self._dimensionality = ImmutableDimensionality(units)

    @property
    def dimensionality(self):
        return copy.copy(self._dimensionality)

    @property
    def magnitude(self):
        return self.view(type=numpy.ndarray)

    @property
    def units(self):
        return str(self.dimensionality)

    def __array_finalize__(self, obj):
        self._dimensionality = getattr(
            obj, 'dimensionality', MutableDimensionality()
        )
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
            assert isinstance(other, Quantity)
        dims = self.dimensionality + other.dimensionality
        magnitude = self.magnitude + other.magnitude
        return Quantity(magnitude, dims, magnitude.dtype)

    def __sub__(self, other):
        if self.dimensionality:
            assert isinstance(other, Quantity)
        dims = self.dimensionality - other.dimensionality
        magnitude = self.magnitude - other.magnitude
        return Quantity(magnitude, dims, magnitude.dtype)

    def __mul__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        try:
            dims = self.dimensionality * other.dimensionality
            magnitude = self.magnitude * other.magnitude
        except:
            dims = copy.copy(self.dimensionality)
            magnitude = self.magnitude * other
        return Quantity(magnitude, dims, magnitude.dtype)

    def __truediv__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        try:
            dims = self.dimensionality / other.dimensionality
            magnitude = self.magnitude / other.magnitude
        except:
            dims = copy.copy(self.dimensionality)
            magnitude = self.magnitude / other
        return Quantity(magnitude, dims, magnitude.dtype)

    __div__ = __truediv__

    def __rmul__(self, other):
        # TODO: This needs to be properly implemented
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return other * self**-1

    __rdiv__ = __rtruediv__

    def __pow__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        dims = self.dimensionality**other
        magnitude = self.magnitude**other
        return Quantity(magnitude, dims, magnitude.dtype)

    def __repr__(self):
        return '%s*%s'%(numpy.ndarray.__str__(self), self.units)

    __str__ = __repr__
