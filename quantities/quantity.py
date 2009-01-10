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
            %(s.units, o.units)
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

    def __new__(cls, data, units='', dtype='d', mutable=True, copy=True):
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data, dtype=dtype)

        if copy == True:
            data = data.copy()

        if isinstance(data, Quantity) and units:
            if isinstance(units, BaseDimensionality):
                units = str(units)
            data = data.rescale(units)

        # should this be a "cooperative super" call instead?
        ret = numpy.ndarray.__new__(
            cls,
            data.shape,
            data.dtype,
            buffer=data
        )
        ret.flags.writeable = mutable
        return ret

    def __init__(self, data, units='', dtype='d', mutable=True, copy=True):
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
#            self.units
#        )

#    def __cmp__(self, other):
#        raise

    def __add__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, copy=False)

        dims = self.dimensionality + other.dimensionality
        magnitude = self.magnitude + other.magnitude

        return Quantity(magnitude, dims, magnitude.dtype)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, copy=False)

        dims = self.dimensionality - other.dimensionality
        magnitude = self.magnitude - other.magnitude

        return Quantity(magnitude, dims, magnitude.dtype)

    def __rsub__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, copy=False)

        dims = other.dimensionality - self.dimensionality
        magnitude = other.magnitude - self.magnitude

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
        if isinstance(other, Quantity):
            simplified = other.simplified

            if simplified.dimensionality:
                raise ValueError("exponent must be dimensionless")
            other = simplified.magnitude

        assert isinstance(other, (numpy.ndarray, int, float, long))

        dims = self.dimensionality**other
        magnitude = self.magnitude**other
        return Quantity(magnitude, dims, magnitude.dtype)

    def __rpow__(self, other):
        simplified = self.simplified
        if simplified.dimensionality:
            raise ValueError("exponent must be dimensionless")

        return other**simplified.magnitude

    def __repr__(self):
        return '%s*%s'%(numpy.ndarray.__str__(self), self.units)

    __str__ = __repr__

    def __getitem__(self, key):
        return Quantity(self.magnitude[key], self.units)

    def __setitem__(self, key, value):
        if not isinstance(value, Quantity):
            value = Quantity(value)

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

    #I don't think this implementation is particularly efficient,
    #perhaps there is something better
    def tolist(self):
        #first get a dummy array from the ndarray method
        work_list = self.magnitude.tolist()
        #now go through and replace all numbers with the appropriate Quantity
        self._tolist(work_list)
        return work_list

    def _tolist(self, work_list):
        #iterate through all the items in the list
        for i in range(len(work_list)):
            #if it's a list then iterate through that list
            if isinstance(work_list[i], list):
                self._tolist(work_list[i])
            else:
                #if it's a number then replace it
                # with the appropriate quantity
                work_list[i] = Quantity(work_list[i], self.dimensionality)

    #need to implement other Array conversion methods:
    # item, itemset, tofile, dump, astype, byteswap

    def sum(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.sum(axis, dtype, out),
            self.dimensionality,
            copy=False
        )

    def fill(self, scalar):
        # the behavior of fill needs to be discussed in the future
        # particularly the fact that this will raise an error if fill (0)
        #is called (when we self is not dimensionless)

        if not isinstance (scalar, Quantity):
            scalar = Quantity(scalar, copy=False)

        if scalar.dimensionality == self.dimensionality:
            self.magnitude.fill(scalar.magnitude)
        else:
            raise ValueError("scalar must have the same units as self")

    #reshape works as intended
    #transpose works as intended
    #reshape works as intented
    #flatten works as expected
    #ravel works as expected
    #squeeze works as expected
    #take functions as intended

    def put (self, indicies, values, mode='raise'):
        """
        performs the equivalent of ndarray.put() but enforces units
        values - must be an Quantity with the same units as self
        """
        if isinstance(values, Quantity):
            #this currently checks to see if the quantities are identical
            # and may be revised in the future, pending discussion
            if values.dimensionality == self.dimensionality:
                self.magnitude.put(indicies, values, mode)
            else:
                raise ValueError("values must have the same units as self")
        else:
            raise TypeError("values must be a Quantity")

    #repeat performs as expected

    #choose does not function correctly, and it is not clear
    # how it would function, so for now it will not be implemented

    #sort works as intended

    def argsort (self, axis=-1, kind='quick', order=None):
        return self.magnitude.argsort(axis, kind, order)

    def searchsorted(self,values, side='left'):
        # if the input is not a Quantity, convert it
        if not isinstance (values, Quantity):
            values = Quantity(values, copy=False)

        if values.dimensionality != self.dimensionality:
            raise ValueError("values does not have the same units as self")

        return self.magnitude.searchsorted(values.magnitude, side)

    def nonzero(self):
        return self.magnitude.nonzero()

    #compress works as intended
    #diagonal works as intended

    def max(self, axis=None, out=None):
        return Quantity(
            self.magnitude.max(),
            self.dimensionality,
            copy=False
        )

    #argmax works as intended

    def min(self, axis=None, out=None):
        return Quantity(
            self.magnitude.min(),
            self.dimensionality,
            copy=False
        )

    def argmin (self,axis=None, out=None):
        return self.magnitude.argmin()

    def ptp(self, axis=None, out=None):
        return Quantity(
            self.magnitude.ptp(),
            self.dimensionality,
            copy=False
        )

    def clip (self, min = None, max = None, out = None):
        if min is None or  max is None:
            raise ValueError("at least one of min or max must be set")
        else:
            #set min and max to their appropriate values
            if min is None: min = Quantity(-numpy.Inf, self.dimensionality)
            if max is None: max = Quantity(numpy.Inf, self.dimensionality)

        if not isinstance(min, Quantity) or not isinstance(max, Quantity):
            raise TypeError("both min and max must be Quantities")

        clipped = self.magnitude.clip(
            min.rescale(self.dimensionality).magnitude,
            max.rescale(self.dimensionality).magnitude,
            out
        )
        return Quantity(clipped, self.dimensionality, copy=False)

    # conj, and conjugate will not currently be implemented
    # because it is not settled how we want to deal with
    # complex numbers

    def round (self, decimals=0, out=None):
        return Quantity(
            self.magnitude.round(decimals, out),
            self.dimensionality,
            copy=False
        )

    def trace (self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        return Quantity(
            self.magnitude.trace(offset, axis1, axis2, dtype, out),
            self.dimensionality,
            copy=False
        )

    # cumsum works as intended

    def mean(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.mean(axis, dtype, out),
            self.dimensionality,
            copy=False)

    def var(self, axis=None, dtype=None, out=None):
        #just return the variance of the magnitude
        # with the correct units (squared)
        return Quantity(
            self.magnitude.var(axis, dtype, out),
            self.dimensionality**2,
            copy=False
        )

    def std (self, axis=None, dtype=None, out=None):
        #just return the std of the magnitude
        return Quantity(
            self.magnitude.std(axis, dtype, out),
            self.dimensionality,
            copy=False
        )

    def prod(self, axis=None, dtype=None, out=None):
        if axis == None:
            power = self.size
        else:
            power = self.shape[axis]

        return Quantity(
            self.magnitude.prod(axis, dtype, out),
            self.dimensionality**power,
            copy=False
        )

    def cumprod(self, axis=None, dtype=None, out=None):
        # cumprod can only be calculated when the quantity is dimensionless
        # otherwise different array elements would have different dimensionality
        if self.dimensionality:
            raise ValueError(
                "Quantity must be dimensionless, try using simplified"
            )
        else:
            return Quantity(
                self.magnitude.cumprod(axis, dtype, out),
                copy=False
                )

    # for now all() and any() will be left unimplemented because this is
    # ambiguous

    #list of unsupported functions: [all, any conj, conjugate, choose]

    #TODO:
    #check/implement resize, swapaxis


class UncertainQuantity(Quantity):

    # TODO: what is an appropriate value?
    __array_priority__ = 22

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
