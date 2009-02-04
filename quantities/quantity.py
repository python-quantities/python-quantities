"""
"""
from __future__ import absolute_import

import copy

import numpy

from .dimensionality import Dimensionality
from .registry import unit_registry
from .utilities import usedoc


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

def validate_unit_quantity(value):
    try:
        assert isinstance(value, Quantity)
        assert value.shape in ((), (1, ))
        assert value.magnitude == 1
    except AssertionError:
        raise ValueError(
                'units must be a scalar Quantity with unit magnitude, got %s'\
                %value
            )
    return value

def validate_dimensionality(value):
    if isinstance(value, str):
        return unit_registry[value].dimensionality
    elif isinstance(value, Quantity):
        validate_unit_quantity(value)
        return value.dimensionality
    elif isinstance(value, Dimensionality):
        return value.copy()
    else:
        raise TypeError(
            'units must be a quantity, string, or dimensionality, got %s'\
            %type(value)
        )

def get_conversion_factor(from_u, to_u):
    validate_unit_quantity(from_u)
    validate_unit_quantity(to_u)
    from_u = from_u.simplified
    to_u = to_u.simplified
    assert from_u.dimensionality == to_u.dimensionality
    return from_u.magnitude / to_u.magnitude


class Quantity(numpy.ndarray):

    # TODO: what is an appropriate value?
    __array_priority__ = 21

    def __new__(cls, data, units='', dtype=None, copy=True):
        if isinstance(data, cls):
            if units:
                data = data.rescale(units)
            return numpy.array(data, dtype=dtype, copy=copy, subok=True)

        ret = numpy.array(data, dtype=dtype, copy=copy).view(cls)
        ret._dimensionality.update(validate_dimensionality(units))
        return ret

    @property
    def dimensionality(self):
        return self._dimensionality.copy()

    @property
    def magnitude(self):
        return self.view(type=numpy.ndarray)

    @property
    def simplified(self):
        rq = 1*unit_registry['dimensionless']
        for u, d in self.dimensionality.iteritems():
            rq = rq * u.simplified**d
        return rq * self.magnitude

    def _get_units(self):
        return Quantity(1.0, self.dimensionality)
    def _set_units(self, units):
        try:
            assert not isinstance(self.base, Quantity)
        except AssertionError:
            raise ValueError('can not modify units of a view of a Quantity')
        try:
            assert self.flags.writeable
        except AssertionError:
            raise ValueError('array is not writeable')
        to_dims = validate_dimensionality(units)
        if self._dimensionality == to_dims:
            return
        to_u = Quantity(1.0, to_dims)
        from_u = Quantity(1.0, self._dimensionality)
        try:
            cf = get_conversion_factor(from_u, to_u)
        except AssertionError:
            raise ValueError(
                'Unable to convert between units of "%s" and "%s"'
                %(from_u._dimensionality, to_u._dimensionality)
            )
        mag = self.magnitude
        mag *= cf
        self._dimensionality = to_u.dimensionality
    units = property(_get_units, _set_units)

    def rescale(self, units):
        """
        Return a copy of the quantity converted to the specified units
        """
        to_dims = validate_dimensionality(units)
        if self.dimensionality == to_dims:
            return self.astype(None)
        to_u = Quantity(1.0, to_dims)
        from_u = Quantity(1.0, self.dimensionality)
        try:
            cf = get_conversion_factor(from_u, to_u)
        except AssertionError:
            raise ValueError(
                'Unable to convert between units of "%s" and "%s"'
                %(from_u._dimensionality, to_u._dimensionality)
            )
        return Quantity(cf*self.magnitude, to_u)

    @usedoc(
        numpy.ndarray.astype,
        suffix='Scalars are returned as scalar Quantity arrays.'
    )
    def astype(self, dtype=None):
        ret = super(Quantity, self).astype(dtype)
        # scalar quantities get converted to plain numbers, so we fix it
        # might be related to numpy ticket # 826
        if not isinstance(ret, type(self)):
            ret = type(self)(ret, self._dimensionality)

        return ret

    def __array_finalize__(self, obj):
        self._dimensionality = getattr(obj, 'dimensionality', Dimensionality())

    def __array_wrap__(self, obj, context):
        # this is experimental right now, there is probably a better
        # way to implement it, but for now lets identify which
        # ufuncs need to be addressed. Maybe a good way to do this would
        # be something like a dictionary mapping of ufuncs to functions
        # that return a proper dimensionality based on the inputs.
#        print obj, context
        uf, objs, huh = context

        result = obj.view(type(self))
        if uf is numpy.multiply:
            result._dimensionality = objs[0].dimensionality * objs[1].dimensionality
        elif uf is numpy.sqrt:
            result._dimensionality = objs[0].dimensionality**(0.5)
        elif uf is numpy.rint:
            result._dimensionality = objs[0].dimensionality
        elif uf is numpy.conjugate:
            result._dimensionality = objs[0].dimensionality
        return result

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

    @usedoc(numpy.ndarray.__add__)
    def __add__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, copy=False)

        dims = self.dimensionality + other.dimensionality
        ret = super(Quantity, self).__add__(other)
        ret._dimensionality = dims
        return ret

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
            dims = self.dimensionality

        ret = super(Quantity, self).__mul__(other)
        ret._dimensionality = dims
        return ret

    def __imul__(self, other):
        if getattr(other, 'dimensionality', None):
            try:
                assert not isinstance(self.base, Quantity)
            except AssertionError:
                raise ValueError('can not modify units of a view of a Quantity')

        try:
            self._dimensionality *= other.dimensionality
        except AttributeError:
            other = numpy.asarray(other).view(Quantity)

        return super(Quantity, self).__imul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            dims = self.dimensionality / other.dimensionality
        except AttributeError:
            other = numpy.asarray(other).view(Quantity)
            dims = self.dimensionality

        ret = super(Quantity, self).__truediv__(other)
        ret._dimensionality = dims
        return ret

    def __div__(self, other):
        return self.__truediv__(other)

    def __itruediv__(self, other):
        if getattr(other, 'dimensionality', None):
            try:
                assert not isinstance(self.base, Quantity)
            except AssertionError:
                raise ValueError('can not modify units of a view of a Quantity')

        try:
            self._dimensionality /= other.dimensionality
        except AttributeError:
            other = numpy.asarray(other).view(Quantity)

        return super(Quantity, self).__itruediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __rtruediv__(self, other):
        try:
            dims = other.dimensionality / self.dimensionality
        except AttributeError:
            other = numpy.asarray(other).view(Quantity)
            dims = self._dimensionality**-1

        ret = super(Quantity, self).__rtruediv__(other)
        ret._dimensionality = dims
        return ret

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __pow__(self, other):
        if getattr(other, 'dimensionality', None):
            raise ValueError("exponent must be dimensionless")

        other = numpy.asarray(other)
        try:
            assert other.min() == other.max()
        except AssertionError:
            raise ValueError('Quantities must be raised to a single power')

        dims = self._dimensionality**other.min()
        ret = super(Quantity, self).__pow__(other)
        ret._dimensionality = dims
        return ret

    def __ipow__(self, other):
        if getattr(other, 'dimensionality', None):
            try:
                assert not isinstance(self.base, Quantity)
            except AssertionError:
                raise ValueError('can not modify units of a view of a Quantity')

        if getattr(other, 'dimensionality', None):
            raise ValueError("exponent must be dimensionless")

        other = numpy.asarray(other)
        try:
            assert other.min() == other.max()
        except AssertionError:
            raise ValueError('Quantities must be raised to a single power')

        self._dimensionality **= other.min()
        return super(Quantity, self).__ipow__(other)

    def __rpow__(self, other):
        if self.dimensionality.simplified:
            raise ValueError("exponent must be dimensionless")

        return super(Quantity, self.simplified).__rpow__(other)

    def __repr__(self):
        return '%s*%s'%(
            repr(self.magnitude), repr(self.dimensionality)
        )

    def __str__(self):
        return '%s %s'%(
            str(self.magnitude), str(self.dimensionality)
        )

    def __getitem__(self, key):
        if isinstance(key, int):
            # This might be resolved by issue # 826
            return Quantity(self.magnitude[key], self._dimensionality)
        else:
            return super(Quantity, self).__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Quantity):
            value = Quantity(value)

        # TODO: do we want this kind of magic?
        self.magnitude[key] = value.rescale(self._dimensionality).magnitude

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
        for i in range(len(work_list)):
            #if it's a list then iterate through that list
            if isinstance(work_list[i], list):
                self._tolist(work_list[i])
            else:
                #if it's a number then replace it
                # with the appropriate quantity
                work_list[i] = Quantity(work_list[i], self.dimensionality)

    #need to implement other Array conversion methods:
    # item, itemset, tofile, dump, byteswap

    def sum(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.sum(axis, dtype, out),
            self.dimensionality,
            copy=False
        )

    def fill(self, scalar):
        if not isinstance (scalar, Quantity):
            scalar = Quantity(scalar, copy=False)

        if scalar._dimensionality == self._dimensionality:
            self.magnitude.fill(scalar.magnitude)
        else:
            raise ValueError("scalar must have the same units as self")

    def put(self, indicies, values, mode='raise'):
        """
        performs the equivalent of ndarray.put() but enforces units
        values - must be an Quantity with the same units as self
        """
        if isinstance(values, Quantity):
            if values._dimensionality == self._dimensionality:
                self.magnitude.put(indicies, values, mode)
            else:
                raise ValueError("values must have the same units as self")
        else:
            raise TypeError("values must be a Quantity")

    # choose does not function correctly, and it is not clear
    # how it would function, so for now it will not be implemented

    def argsort(self, axis=-1, kind='quick', order=None):
        return self.magnitude.argsort(axis, kind, order)

    def searchsorted(self,values, side='left'):
        if not isinstance (values, Quantity):
            values = Quantity(values, copy=False)

        if values._dimensionality != self._dimensionality:
            raise ValueError("values does not have the same units as self")

        return self.magnitude.searchsorted(values.magnitude, side)

    def nonzero(self):
        return self.magnitude.nonzero()

    def max(self, axis=None, out=None):
        return Quantity(
            self.magnitude.max(),
            self.dimensionality,
            copy=False
        )

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
            if min is None: min = Quantity(-numpy.Inf, self._dimensionality)
            if max is None: max = Quantity(numpy.Inf, self._dimensionality)

        if self.dimensionality and not \
                (isinstance(min, Quantity) and isinstance(max, Quantity)):
            raise ValueError(
                "both min and max must be Quantities with compatible units"
            )

        clipped = self.magnitude.clip(
            min.rescale(self._dimensionality).magnitude,
            max.rescale(self._dimensionality).magnitude,
            out
        )
        return Quantity(clipped, self.dimensionality, copy=False)

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

    def mean(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.mean(axis, dtype, out),
            self.dimensionality,
            copy=False)

    def var(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.var(axis, dtype, out),
            self._dimensionality**2,
            copy=False
        )

    def std(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.std(axis, dtype, out),
            self._dimensionality,
            copy=False
        )

    def prod(self, axis=None, dtype=None, out=None):
        if axis == None:
            power = self.size
        else:
            power = self.shape[axis]

        return Quantity(
            self.magnitude.prod(axis, dtype, out),
            self._dimensionality**power,
            copy=False
        )

    def cumprod(self, axis=None, dtype=None, out=None):
        if self._dimensionality:
            # different array elements would have different dimensionality
            raise ValueError(
                "Quantity must be dimensionless, try using simplified"
            )
        else:
            return Quantity(
                self.magnitude.cumprod(axis, dtype, out),
                copy=False
                )

    # list of unsupported functions: [choose]
