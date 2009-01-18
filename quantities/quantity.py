"""
"""

import copy

import numpy

from quantities.dimensionality import BaseDimensionality, \
    Dimensionality, ImmutableDimensionality
from quantities.registry import unit_registry

def prepare_compatible_units(s, o):
    if not isinstance(o, Quantity):
        o = Quantity(o, copy=False)
    try:
        assert s.dimensionality.simplified == o.dimensionality.simplified
        return s.simplified, o.simplified
    except AssertionError:
        raise ValueError(
            'can not compare quantities with units of %s and %s'\
            %(s.units, o.units)
        )


class QuantityIterator:

    """an iterator for quantity objects"""

    def __init__(self, object):
        self.object = object
        self._iterator = super(Quantity, object).__iter__()

    def next(self):
        return Quantity(self._iterator.next(), self.object.units)


class Quantity(numpy.ndarray):

    # TODO: what is an appropriate value?
    __array_priority__ = 21

    def __new__(cls, data, units='', dtype='d', copy=True):
        if isinstance(data, Quantity):
            if units:
                # force a copy so we don't rescale a subset of the original
                copy = True

            res = numpy.array(data, dtype=dtype, copy=copy).view(cls)
            if copy:
                res._dimensionality = data._dimensionality.copy()
            else:
                res._dimensionality = data._dimensionality
            if units:
                res.units = units

            return res

        res = numpy.array(data, dtype=dtype, copy=copy).view(cls)

        if isinstance(units, str):
            if units in ('', 'dimensionless'):
                dims = {}
            else:
                dims = unit_registry[units].dimensionality
        elif isinstance(units, Quantity):
            dims = units.dimensionality
        elif isinstance(units, (BaseDimensionality, dict)):
            dims = units
        else:
            raise TypeError(
                'units must be a quantity, string, or dimensionality, got %s'\
                %type(units)
            )
        res._dimensionality = Dimensionality(dims)

        return res

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def magnitude(self):
        return self.view(type=numpy.ndarray)

    def get_units(self):
        return Quantity(1, self.dimensionality)
    def set_units(self, other):
        if not self.flags.writeable:
            raise AttributeError("can not modify protected data")
        if isinstance(other, str):
            other = unit_registry[other]
        if isinstance(other, BaseDimensionality):
            other = Quantity(1, other)
        if isinstance(other, Quantity):
            try:
                assert other.magnitude == 1
            except AssertionError:
                raise ValueError('units must have unit magnitude')
        try:
            sq = Quantity(1.0, self.dimensionality).simplified
            osq = other.simplified
            assert osq.dimensionality == sq.dimensionality
            m = self.magnitude
            m *= sq.magnitude / osq.magnitude
            self._dimensionality = \
                Dimensionality(other.dimensionality)
        except AssertionError:
            raise ValueError(
                'Unable to convert between units of "%s" and "%s"'
                %(sq.units, osq.units)
            )
    units = property(get_units, set_units)

    def rescale(self, units):
        """
        Return a copy of the quantity converted to the specified units
        """
        return Quantity(self, units)

    @property
    def simplified(self):
        rq = 1*unit_registry['dimensionless']
        for u, d in self.dimensionality.iteritems():
            rq = rq * u.reference_quantity**d
        return rq * self.magnitude

    def __array_finalize__(self, obj):
        self._dimensionality = getattr(obj, '_dimensionality', Dimensionality())
        if self.base is None:
            self._dimensionality = self._dimensionality.copy()

#    def __array_wrap__(self, obj, context=None):
#        """
#        Special hook for ufuncs.
#        Wraps the numpy array and sets the mask according to context.
#        """
#        result = obj.view(type(self))
#
#        if context is not None:
#            result._dimensionality = result._dimensionality.copy()
#            (func, args, _) = context
#            m = reduce(mask_or, [getmaskarray(arg) for arg in args])
#            # Get the domain mask................
#            domain = ufunc_domain.get(func, None)
#            if domain is not None:
#                if len(args) > 2:
#                    d = reduce(domain, args)
#                else:
#                    d = domain(*args)
#                # Fill the result where the domain is wrong
#                try:
#                    # Binary domain: take the last value
#                    fill_value = ufunc_fills[func][-1]
#                except TypeError:
#                    # Unary domain: just use this one
#                    fill_value = ufunc_fills[func]
#                except KeyError:
#                    # Domain not recognized, use fill_value instead
#                    fill_value = self.fill_value
#                result = result.copy()
#                np.putmask(result, d, fill_value)
#                # Update the mask
#                if m is nomask:
#                    if d is not nomask:
#                        m = d
#                else:
#                    m |= d
#            # Make sure the mask has the proper size
#            if result.shape == () and m:
#                return masked
#            else:
#                result._mask = m
#                result._sharedmask = False
#        #....
#        return result

    def __add__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, copy=False)

        dims = self.dimensionality + other.dimensionality
        ret = super(Quantity, self).__add__(other)
        ret._dimensionality = dims
        return ret

    # TODO: in-place arithmetic should check for .base, and raise if not None

    def __iadd__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, copy=False)

        self._dimensionality += other.dimensionality
        return super(Quantity, self).__iadd__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            other = numpy.asarray(other).view(Quantity)

        dims = self.dimensionality - other.dimensionality
        ret = super(Quantity, self).__sub__(other)
        ret._dimensionality = dims
        return ret

    def __isub__(self, other):
        if not isinstance(other, Quantity):
            other = numpy.asarray(other).view(Quantity)

        self._dimensionality -= other.dimensionality
        return super(Quantity, self).__isub__(other)

    def __rsub__(self, other):
        if not isinstance(other, Quantity):
            other = numpy.asarray(other).view(Quantity)

        dims = other.dimensionality - self.dimensionality
        ret = super(Quantity, self).__rsub__(other)
        ret._dimensionality = dims
        return ret

    def __mul__(self, other):
        try:
            dims = self.dimensionality * other.dimensionality
        except AttributeError:
            other = numpy.asarray(other).view(Quantity)
            dims = Dimensionality(self.dimensionality)

        ret = super(Quantity, self).__mul__(other)
        ret._dimensionality = dims
        return ret

    def __imul__(self, other):
        try:
            self._dimensionality *= other.dimensionality
        except AttributeError:
            pass

        return super(Quantity, self).__imul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            dims = self.dimensionality / other.dimensionality
        except AttributeError:
            other = numpy.asarray(other).view(Quantity)
            dims = Dimensionality(self.dimensionality)

        ret = super(Quantity, self).__truediv__(other)
        ret._dimensionality = dims
        return ret

    def __div__(self, other):
        return self.__truediv__(other)

    def __itruediv__(self, other):
        try:
            self._dimensionality /= other.dimensionality
        except AttributeError:
            pass
        return super(Quantity, self).__itruediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __rtruediv__(self, other):
        try:
            dims = other.dimensionality / self.dimensionality
        except AttributeError:
            other = numpy.asarray(other).view(Quantity)
            dims = Dimensionality(self.dimensionality**-1)

        ret = super(Quantity, self).__rtruediv__(other)
        ret._dimensionality = dims
        return ret

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __pow__(self, other):
        if isinstance(other, Quantity):
            if other.dimensionality.simplified:
                raise ValueError("exponent must be dimensionless")
            other = other.simplified.magnitude

        other = numpy.asarray(other)
        try:
            assert other.min() == other.max()
            other = other.min()
        except AssertionError:
            raise ValueError('Quantities must be raised to a single power')

        dims = self.dimensionality**other
        ret = super(Quantity, self).__pow__(other)
        ret._dimensionality = dims
        return ret

    def __ipow__(self, other):
        if isinstance(other, Quantity):
            if other.dimensionality.simplified:
                raise ValueError("exponent must be dimensionless")
            other = other.simplified.magnitude

        other = numpy.asarray(other)
        try:
            assert other.min() == other.max()
            other = other.min()
        except AssertionError:
            raise ValueError('Quantities must be raised to a single power')

        self._dimensionality **= other
        return super(Quantity, self).__ipow__(other)

    def __rpow__(self, other):
        if self.dimensionality.simplified:
            raise ValueError("exponent must be dimensionless")

        return super(Quantity, self.simplified).__rpow__(other)

    def __repr__(self):
        return '%s %s'%(numpy.ndarray.__str__(self), self.dimensionality)

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
        if not isinstance (scalar, Quantity):
            scalar = Quantity(scalar, copy=False)

        if scalar.dimensionality == self.dimensionality:
            self.magnitude.fill(scalar.magnitude)
        else:
            raise ValueError("scalar must have the same units as self")

    #reshape works as intended
    #transpose works as intended
    #reshape works as intended
    #flatten works as expected
    #ravel works as expected
    #squeeze works as expected
    #take functions as intended

    def put(self, indicies, values, mode='raise'):
        """
        performs the equivalent of ndarray.put() but enforces units
        values - must be an Quantity with the same units as self
        """
        if isinstance(values, Quantity):
            if values.dimensionality == self.dimensionality:
                self.magnitude.put(indicies, values, mode)
            else:
                raise ValueError("values must have the same units as self")
        else:
            raise TypeError("values must be a Quantity")

    # repeat performs as expected

    # choose does not function correctly, and it is not clear
    # how it would function, so for now it will not be implemented

    # sort works as intended

    def argsort(self, axis=-1, kind='quick', order=None):
        return self.magnitude.argsort(axis, kind, order)

    def searchsorted(self,values, side='left'):
        if not isinstance (values, Quantity):
            values = Quantity(values, copy=False)

        if values.dimensionality != self.dimensionality:
            raise ValueError("values does not have the same units as self")

        return self.magnitude.searchsorted(values.magnitude, side)

    def nonzero(self):
        return self.magnitude.nonzero()

    # compress works as intended
    # diagonal works as intended

    def max(self, axis=None, out=None):
        return Quantity(
            self.magnitude.max(),
            self.dimensionality,
            copy=False
        )

    # argmax works as intended

    def min(self, axis=None, out=None):
        return Quantity(
            self.magnitude.min(),
            self.dimensionality,
            copy=False
        )

    def argmin(self,axis=None, out=None):
        return self.magnitude.argmin()

    def ptp(self, axis=None, out=None):
        return Quantity(
            self.magnitude.ptp(),
            self.dimensionality,
            copy=False
        )

    def clip(self, min=None, max=None, out=None):
        if min is None and max is None:
            raise ValueError("at least one of min or max must be set")
        else:
            if min is None: min = Quantity(-numpy.Inf, self.dimensionality)
            if max is None: max = Quantity(numpy.Inf, self.dimensionality)

        if self.dimensionality and not \
                (isinstance(min, Quantity) and isinstance(max, Quantity)):
            raise ValueError(
                "both min and max must be Quantities with compatible units"
            )

        clipped = self.magnitude.clip(
            min.rescale(self.units).magnitude,
            max.rescale(self.units).magnitude,
            out
        )
        return Quantity(clipped, self.dimensionality, copy=False)

    # conj, and conjugate will not currently be implemented
    # because it is not settled how we want to deal with
    # complex numbers

    def round(self, decimals=0, out=None):
        return Quantity(
            self.magnitude.round(decimals, out),
            self.dimensionality,
            copy=False
        )

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
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
        return Quantity(
            self.magnitude.var(axis, dtype, out),
            self.dimensionality**2,
            copy=False
        )

    def std(self, axis=None, dtype=None, out=None):
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
        if self.dimensionality:
            # different array elements would have different dimensionality
            raise ValueError(
                "Quantity must be dimensionless, try using simplified"
            )
        else:
            return Quantity(
                self.magnitude.cumprod(axis, dtype, out),
                copy=False
                )

    # list of unsupported functions: [conj, conjugate, choose]
