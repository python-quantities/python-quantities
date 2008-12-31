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
    
#    def to(self, units):
#        """this function returns a copy of the object with the specified units
#        """
#        copy = self.__deepcopy__()
#        copy.units = units
#        return copy

    def __getitem__(self, key):
        """
        returns a quantity
        """
        # indexing needs overloading so that units are also returned
        data = self.view(type=ndarray)[key]
        return Quantity(data, self.units)

    def __setitem__(self, key, value):
        ## convert value units to item's units
        if (self.units != value.units):
            #this can be replaced with .to()
            value = value.__deepcopy__()
            value.units = self.units

        self.view(dtype = ndarray).__setitem__(key, value)

    def __iter__(self):
        # return the iterator wrapper
        return QuantityIterator(self)


    def _comparison_operater_prep(self, other):
        """
        this function checks whether other is of an appropriate type and returns an ndarray
        object which is other modified so that it is in the correct units and scaling factor
        other - the other object to be operated with and
        returns: (prepped_other)
        """
        if (not isinstance(other, Quantity)):
            other = Quantity(other, '')

        # this can be replaced with .to()
        other = other.__deepcopy__()
        other.units = self.units
        return self.view(type=ndarray), other.view(type=ndarray)

    # comparison overloads
    # these must be implemented so that the return type is just a plain array
    # (no units) and so that the proper scaling is used
    # these comparisons work even though self will be Quantity and other will be
    # a ndarray (after going though _comparison_operater_prep) because we use
    # the ndarray comparison operators and those naturally disregard the effect
    # of the units
    def __lt__(self, other):

       self, other = self._comparison_operater_prep(other)

       return self.__lt__(other)


    def __le__(self, other):
       self, other = self._comparison_operater_prep(other)

       return self.__le__(other)

    def __eq__(self, other):
       self, other = self._comparison_operater_prep(other)

       return self.__eq__(other)

    def __ne__(self, other):
       self, other = self._comparison_operater_prep(other)

       return self.__ne__(other)

    def __gt__(self, other):
       self, other = self._comparison_operater_prep(other)

       return self.__gt__(other)

    def __ge__(self, other):
       self, other = self._comparison_operater_prep(other)

       return self.__ge__(other)

#define an iterator class
class QuantityIterator:
    """ an iterator for quantity objects"""
    # this simply wraps the base class iterator

    def __init__(self, object):
        """mu"""
        self.object = object
        self.iterator = super(Quantity, object).__iter__()

    def __iter__(self):
        return self

    def next(self):
        # we just want to return the ndarray item times the units
        return Quantity(self.iterator.next(), self.object.units)
