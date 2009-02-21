"""
"""
from __future__ import absolute_import

import copy
from functools import wraps

import numpy

from .config import USE_UNICODE
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .utilities import with_doc


def prepare_compatible_units(s, o):
    if not isinstance(o, Quantity):
        o = Quantity(o, copy=False)
    try:
        assert s.dimensionality.simplified == o.dimensionality.simplified
        return s._reference, o._reference
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
    from_u = from_u._reference
    to_u = to_u._reference
    assert from_u.dimensionality == to_u.dimensionality
    return from_u.magnitude / to_u.magnitude

def ensure_quantity(f):
    @wraps(f)
    def g(*args):
        args = list(args)
        if not isinstance(args[1], type(args[0])):
            args[1] = Quantity(args[1], copy=False)
        elif not isinstance(args[0], type(args[1])):
            args[0] = Quantity(args[0], copy=False)
        return f(*args)
    return g


def protected_addition(f):
    @wraps(f)
    def g(self, other, *args):
        if not isinstance(other, Quantity):
            other = Quantity(other, copy=False)
        getattr(self._dimensionality, f.__name__)(other._dimensionality)
        return f(self, other, *args)
    return g

def protected_multiplication(f):
    @wraps(f)
    def g(self, other, *args):
        if getattr(other, 'dimensionality', None):
            try:
                assert not isinstance(self.base, Quantity)
            except AssertionError:
                raise ValueError('can not modify units of a view of a Quantity')
        return f(self, other, *args)
    return g

def check_uniform(f):
    @wraps(f)
    def g(self, other, *args):
        if getattr(other, 'dimensionality', None):
            raise ValueError("exponent must be dimensionless")
        other = numpy.asarray(other)
        try:
            assert other.min() == other.max()
        except AssertionError:
            raise ValueError('Quantities must be raised to a uniform power')
        return f(self, other, *args)
    return g

def protected_power(f):
    @wraps(f)
    def g(self, other, *args):
        if other != 1:
            try:
                assert not isinstance(self.base, Quantity)
            except AssertionError:
                raise ValueError('can not modify units of a view of a Quantity')
        return f(self, other, *args)
    return g


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
    def _reference(self):
        """The reference quantity used to perform conversions"""
        rq = 1*unit_registry['dimensionless']
        for u, d in self.dimensionality.iteritems():
            rq = rq * u._reference**d
        return rq * self.magnitude

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

    @with_doc(numpy.ndarray.astype)
    def astype(self, dtype=None):
        '''Scalars are returned as scalar Quantity arrays.'''
        ret = super(Quantity, self).astype(dtype)
        # scalar quantities get converted to plain numbers, so we fix it
        # might be related to numpy ticket # 826
        if not isinstance(ret, type(self)):
            ret = type(self)(ret, self._dimensionality)

        return ret

    def __array_finalize__(self, obj):
        self._dimensionality = getattr(obj, 'dimensionality', Dimensionality())

    def __array_wrap__(self, obj, context=None):
        if self.__array_priority__ >= Quantity.__array_priority__:
            result = obj.view(type(self))
        else:
            # don't want a UnitQuantity
            result = obj.view(Quantity)
        if context is None:
            return result

        uf, objs, huh = context
#        print obj, uf, objs
        try:
            result._dimensionality = p_dict[uf](*objs)
        except KeyError:
            print 'ufunc %r not implemented, please file a bug report' % uf
        return result

    @with_doc(numpy.ndarray.__add__)
    @protected_addition
    def __add__(self, other):
        return super(Quantity, self).__add__(other)

    @with_doc(numpy.ndarray.__iadd__)
    @protected_addition
    def __iadd__(self, other):
        return super(Quantity, self).__iadd__(other)

    @with_doc(numpy.ndarray.__radd__)
    @protected_addition
    def __radd__(self, other):
        return self.__add__(other)

    @with_doc(numpy.ndarray.__sub__)
    @protected_addition
    def __sub__(self, other):
        return super(Quantity, self).__sub__(other)

    @with_doc(numpy.ndarray.__isub__)
    @protected_addition
    def __isub__(self, other):
        return super(Quantity, self).__isub__(other)

    @with_doc(numpy.ndarray.__rsub__)
    @protected_addition
    def __rsub__(self, other):
        return super(Quantity, self).__rsub__(other)

    @with_doc(numpy.ndarray.__imul__)
    @protected_multiplication
    def __imul__(self, other):
        return super(Quantity, self).__imul__(other)

    @with_doc(numpy.ndarray.__itruediv__)
    @protected_multiplication
    def __itruediv__(self, other):
        return super(Quantity, self).__itruediv__(other)

    @with_doc(numpy.ndarray.__idiv__)
    @protected_multiplication
    def __idiv__(self, other):
        return super(Quantity, self).__itruediv__(other)

    @with_doc(numpy.ndarray.__pow__)
    @check_uniform
    def __pow__(self, other):
        return super(Quantity, self).__pow__(other)

    @with_doc(numpy.ndarray.__ipow__)
    @check_uniform
    @protected_power
    def __ipow__(self, other):
        return super(Quantity, self).__ipow__(other)

    @with_doc(numpy.ndarray.__repr__)
    def __repr__(self):
        return '%s * %s'%(
            repr(self.magnitude), self.dimensionality.string
        )

    @with_doc(numpy.ndarray.__str__)
    def __str__(self):
        if USE_UNICODE:
            dims = self.dimensionality.unicode
        else:
            dims = self.dimensionality.string
        return '%s %s'%(str(self.magnitude), dims)

    @with_doc(numpy.ndarray.__getitem__)
    def __getitem__(self, key):
        if isinstance(key, int):
            # This might be resolved by issue # 826
            return Quantity(self.magnitude[key], self._dimensionality)
        else:
            return super(Quantity, self).__getitem__(key)

    @with_doc(numpy.ndarray.__setitem__)
    def __setitem__(self, key, value):
        if not isinstance(value, Quantity):
            value = Quantity(value)

        # TODO: do we want this kind of magic?
        self.magnitude[key] = value.rescale(self._dimensionality).magnitude

    @with_doc(numpy.ndarray.__lt__)
    def __lt__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude < os.magnitude

    @with_doc(numpy.ndarray.__le__)
    def __le__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude <= os.magnitude

    @with_doc(numpy.ndarray.__eq__)
    def __eq__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude == os.magnitude

    @with_doc(numpy.ndarray.__ne__)
    def __ne__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude != os.magnitude

    @with_doc(numpy.ndarray.__gt__)
    def __gt__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude > os.magnitude

    @with_doc(numpy.ndarray.__ge__)
    def __ge__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude >= os.magnitude

    #I don't think this implementation is particularly efficient,
    #perhaps there is something better
    @with_doc(numpy.ndarray.tolist)
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

    @with_doc(numpy.ndarray.sum)
    def sum(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.sum(axis, dtype, out),
            self.dimensionality,
            copy=False
        )

    @with_doc(numpy.ndarray.fill)
    def fill(self, scalar):
        if not isinstance (scalar, Quantity):
            scalar = Quantity(scalar, copy=False)

        if scalar._dimensionality == self._dimensionality:
            self.magnitude.fill(scalar.magnitude)
        else:
            raise ValueError("scalar must have the same units as self")

    @with_doc(numpy.ndarray.put)
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

    @with_doc(numpy.ndarray.argsort)
    def argsort(self, axis=-1, kind='quick', order=None):
        return self.magnitude.argsort(axis, kind, order)

    @with_doc(numpy.ndarray.searchsorted)
    def searchsorted(self,values, side='left'):
        if not isinstance (values, Quantity):
            values = Quantity(values, copy=False)

        if values._dimensionality != self._dimensionality:
            raise ValueError("values does not have the same units as self")

        return self.magnitude.searchsorted(values.magnitude, side)

    @with_doc(numpy.ndarray.nonzero)
    def nonzero(self):
        return self.magnitude.nonzero()

    @with_doc(numpy.ndarray.max)
    def max(self, axis=None, out=None):
        return Quantity(
            self.magnitude.max(),
            self.dimensionality,
            copy=False
        )

    @with_doc(numpy.ndarray.min)
    def min(self, axis=None, out=None):
        return Quantity(
            self.magnitude.min(),
            self.dimensionality,
            copy=False
        )

    @with_doc(numpy.ndarray.argmin)
    def argmin(self,axis=None, out=None):
        return self.magnitude.argmin()

    @with_doc(numpy.ndarray.ptp)
    def ptp(self, axis=None, out=None):
        return Quantity(
            self.magnitude.ptp(),
            self.dimensionality,
            copy=False
        )

    @with_doc(numpy.ndarray.clip)
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

    @with_doc(numpy.ndarray.round)
    def round(self, decimals=0, out=None):
        return Quantity(
            self.magnitude.round(decimals, out),
            self.dimensionality,
            copy=False
        )

    @with_doc(numpy.ndarray.trace)
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        return Quantity(
            self.magnitude.trace(offset, axis1, axis2, dtype, out),
            self.dimensionality,
            copy=False
        )

    @with_doc(numpy.ndarray.mean)
    def mean(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.mean(axis, dtype, out),
            self.dimensionality,
            copy=False)

    @with_doc(numpy.ndarray.var)
    def var(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.var(axis, dtype, out),
            self._dimensionality**2,
            copy=False
        )

    @with_doc(numpy.ndarray.std)
    def std(self, axis=None, dtype=None, out=None):
        return Quantity(
            self.magnitude.std(axis, dtype, out),
            self._dimensionality,
            copy=False
        )

    @with_doc(numpy.ndarray.prod)
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

    @with_doc(numpy.ndarray.cumprod)
    def cumprod(self, axis=None, dtype=None, out=None):
        if self._dimensionality:
            # different array elements would have different dimensionality
            raise ValueError(
                "Quantity must be dimensionless, try using simplified"
            )
        else:
            return super(Quantity, self).cumprod(axis, dtype, out)

    # list of unsupported functions: [choose]
