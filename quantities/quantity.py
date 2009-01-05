"""
"""

import copy

import numpy

from quantities.dimensionality import BaseDimensionality, \
    MutableDimensionality, ImmutableDimensionality
from quantities.registry import unit_registry

def prepare_compatible_units(s, o):
    try:
        ss, os = s.simplified, o.simplified
        assert ss.units == os.units
        return ss, os
    except AssertionError:
        raise ValueError(
            'can not compare quantities with units of %s and %s'\
            %(ss.units, os.units)
        )


class QuantityIterator:

    """an iterator for quantity objects"""

    def __init__(self, object):
        self.object = object
        self.iterator = super(Quantity, object).__iter__()

    def __iter__(self):
        return self

    def next(self):
        return Quantity(self.iterator.next(), self.object.units)


class Quantity(numpy.ndarray):

    # TODO: what is an appropriate value?
    __array_priority__ = 21

    def __new__(cls, data, units='', dtype='d', mutable=True):
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data, dtype=dtype)

        data = data.copy()

        if isinstance(data, Quantity) and units:
            if isinstance(units, BaseDimensionality):
                units = str(units)
            data = data.rescale(units)

        ret = numpy.ndarray.__new__(
            cls,
            data.shape,
            data.dtype,
            buffer=data
        )
        ret.flags.writeable = mutable
        return ret

    def __init__(self, data, units='', dtype='d', mutable=True):
        if isinstance(data, Quantity) and not units:
            dims = data.dimensionality
        elif isinstance(units, str):
            if units in ('', 'dimensionless'):
                dims = {}
            else:
                dims = unit_registry[units].dimensionality
        elif isinstance(units, Quantity):
            dims = units.dimensionality
        elif isinstance(units, (BaseDimensionality, dict)):
            dims = units
        else:
            assert units is None
            dims = None

        self._mutable = mutable
        if self.is_mutable:
            if dims is None: dims = {}
            self._dimensionality = MutableDimensionality(dims)
        else:
            if dims is None:
                self._dimensionality = None
            else:
                self._dimensionality = ImmutableDimensionality(dims)

    @property
    def dimensionality(self):
        if self._dimensionality is None:
            return ImmutableDimensionality({self:1})
        else:
            return ImmutableDimensionality(self._dimensionality)

    @property
    def magnitude(self):
        return self.view(type=numpy.ndarray)

    @property
    def is_mutable(self):
        return self._mutable

    # get and set methods for the units property
    def get_units(self):
        return str(self.dimensionality)
    def set_units(self, units):
        if not self.is_mutable:
            raise AttributeError("can not modify protected units")
        if isinstance(units, str):
            units = unit_registry[units]
        if isinstance(units, Quantity):
            try:
                assert units.magnitude == 1
            except AssertionError:
                raise ValueError('units must have unit magnitude')
        try:
            sq = Quantity(1.0, self.dimensionality).simplified
            osq = units.simplified
            assert osq.dimensionality == sq.dimensionality
            m = self.magnitude
            m *= sq.magnitude / osq.magnitude
            self._dimensionality = \
                MutableDimensionality(units.dimensionality)
        except AssertionError:
            raise ValueError(
                'Unable to convert between units of "%s" and "%s"'
                %(sq.units, osq.units)
            )
    units = property(get_units, set_units)

    def mean(self):
        return Quantity(self.magnitude.mean(), self.units)

    def rescale(self, units):
        """
        Return a copy of the quantity converted to the specified units
        """
        copy = Quantity(self)
        copy.units = units
        return copy

    @property
    def simplified(self):
        rq = self.magnitude * unit_registry['dimensionless']
        for u, d in self.dimensionality.iteritems():
            rq = rq * u.reference_quantity**d
        return rq

    def __array_finalize__(self, obj):
        self._dimensionality = getattr(
            obj, 'dimensionality', MutableDimensionality()
        )

#    def __deepcopy__(self, memo={}):
#        return self.__class__(
#            self.view(type=numpy.ndarray),
#            self.dtype,
#            self.units,
#            self._uncertainty
#        )

#    def __cmp__(self, other):
#        raise

    def __add__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other)
        dims = self.dimensionality + other.dimensionality
        magnitude = self.magnitude + other.magnitude
        return Quantity(magnitude, dims, magnitude.dtype)

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other)
        dims = self.dimensionality - other.dimensionality
        magnitude = self.magnitude - other.magnitude
        return Quantity(magnitude, dims, magnitude.dtype)

    def __mul__(self, other):
        try:
            dims = self.dimensionality * other.dimensionality
            magnitude = self.magnitude * other.magnitude
        except AttributeError:
            magnitude = self.magnitude * other
            dims = copy.copy(self.dimensionality)
        return Quantity(magnitude, dims, magnitude.dtype)

    def __truediv__(self, other):
        try:
            dims = self.dimensionality / other.dimensionality
            magnitude = self.magnitude / other.magnitude
        except AttributeError:
            magnitude = self.magnitude / other
            dims = copy.copy(self.dimensionality)
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

    def __getitem__(self, key):
        return Quantity(self.magnitude[key], self.units)

    def __setitem__(self, key, value):
        # TODO: do we want this kind of magic?
        self.magnitude[key] = value.rescale(self.units).magnitude

    def __iter__(self):
        return QuantityIterator(self)

    def __lt__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude < os.magnitude

    def __le__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude <= os.magnitude

    def __eq__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude == os.magnitude

    def __ne__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude != os.magnitude

    def __gt__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude > os.magnitude

    def __ge__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude >= os.magnitude


class UncertainQuantity(Quantity):

    # TODO: what is an appropriate value?
    __array_priority__ = 21

    def __new__(
        cls, data, units='', uncertainty=0, dtype='d', mutable=True
    ):
        return Quantity.__new__(
            cls, data, units, dtype, mutable
        )

    def __init__(
        self, data, units='', uncertainty=0, dtype='d', mutable=True
    ):
        Quantity.__init__(
            self, data, units, dtype, mutable
        )
        if not numpy.any(uncertainty):
            uncertainty = getattr(self, 'uncertainty', uncertainty)
        self.set_uncertainty(uncertainty)

    @property
    def simplified(self):
        sq = self.magnitude * unit_registry['dimensionless']
        for u, d in self.dimensionality.iteritems():
            sq = sq * u.reference_quantity**d
        u = self.uncertainty.simplified
        # TODO: use view:
        return UncertainQuantity(sq, uncertainty=u)

    def set_units(self, units):
        Quantity.set_units(self, units)
        self.uncertainty.set_units(units)
    units = property(Quantity.get_units, set_units)

    def get_uncertainty(self):
        return self._uncertainty
    def set_uncertainty(self, uncertainty):
        if not isinstance(uncertainty, Quantity):
            uncertainty = Quantity(uncertainty, self.units)
        try:
            if len(uncertainty.shape) != 0:
                # make sure we can calculate relative uncertainty:
                uncertainty.magnitude / self.magnitude
            uncertainty.units = self.units
            self._uncertainty = uncertainty
        except:
            ValueError(
                'uncertainty must be divisible by the parent quantity'
            )
    uncertainty = property(get_uncertainty, set_uncertainty)

    @property
    def relative_uncertainty(self):
        if len(self.uncertainty.shape) == 0:
            return self.uncertainty.magnitude/self.magnitude.mean()
        return self.uncertainty.magnitude/self.magnitude

    def rescale(self, units):
        """
        Return a copy of the quantity converted to the specified units
        """
        copy = UncertainQuantity(self)
        copy.units = units
        return copy

    def __array_finalize__(self, obj):
        Quantity.__array_finalize__(self, obj)
        self._uncertainty = getattr(
            obj, 'uncertainty', Quantity(0, self.units)
        )

    def __add__(self, other):
        res = Quantity.__add__(self, other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u)

    def __sub__(self, other):
        res = Quantity.__sub__(self, other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u)

    def __mul__(self, other):
        res = Quantity.__mul__(self, other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            if len(ru.shape) == 0:
                u = res.mean() * ru
            else:
                u = res * ru
        except AttributeError:
            u = (self.uncertainty**2*other**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u)

    def __truediv__(self, other):
        res = Quantity.__truediv__(self, other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            if len(ru.shape) == 0:
                u = res.mean() * ru
            else:
                u = res * ru
        except AttributeError:
            u = (self.uncertainty**2/other**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u)

    def __pow__(self, other):
        res = Quantity.__pow__(self, other)
        ru = other * self.relative_uncertainty
        if len(ru.shape) == 0:
            u = res.mean() * ru
        else:
            u = res * ru
        return UncertainQuantity(res, uncertainty=u)

    def __getitem__(self, key):
        return UncertainQuantity(
            self.magnitude[key],
            self.units,
            copy.copy(self.uncertainty)
        )

    def __repr__(self):
        return '%s*%s\n+/-%s (1 sigma)'\
            %(numpy.ndarray.__str__(self), self.units, self.uncertainty)

    __str__ = __repr__
