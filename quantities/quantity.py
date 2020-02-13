"""
"""
from __future__ import absolute_import

import copy
from functools import wraps
import sys

import numpy as np

from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc

PREFERRED = []  # List of preferred quanitities for each symbol,
                # e.g. PREFERRED = [pq.mV, pq.pA, pq.UnitQuantity('femtocoulomb', 1e-15*pq.C, 'fC')]
                # Intended to be overwritten in down-stream packages

if sys.version.startswith('3'):
    unicode = str

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
    if isinstance(value, (str, unicode)):
        try:
            return unit_registry[value].dimensionality
        except (KeyError, UnicodeDecodeError):
            return unit_registry[str(value)].dimensionality
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

def scale_other_units(f):
    @wraps(f)
    def g(self, other, *args):
        other = np.asanyarray(other)
        if not isinstance(other, Quantity):
            other = other.view(type=Quantity)
        if other._dimensionality != self._dimensionality:
            other = other.rescale(self.units)
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
        other = np.asarray(other)
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

def wrap_comparison(f):
    @wraps(f)
    def g(self, other):
        if isinstance(other, Quantity):
            if other._dimensionality != self._dimensionality:
                other = other.rescale(self._dimensionality)
            other = other.magnitude
        return f(self, other)
    return g


class Quantity(np.ndarray):

    # TODO: what is an appropriate value?
    __array_priority__ = 21

    def __new__(cls, data, units='', dtype=None, copy=True):
        if isinstance(data, Quantity):
            if units:
                data = data.rescale(units)
            if isinstance(data, unit_registry['UnitQuantity']):
                return 1*data
            return np.array(data, dtype=dtype, copy=copy, subok=True).view(cls)

        ret = np.array(data, dtype=dtype, copy=copy).view(cls)
        ret._dimensionality.update(validate_dimensionality(units))
        return ret

    @property
    def dimensionality(self):
        return self._dimensionality.copy()

    @property
    def _reference(self):
        """The reference quantity used to perform conversions"""
        rq = 1*unit_registry['dimensionless']
        for u, d in self.dimensionality.items():
            rq = rq * u._reference**d
        return rq * self.magnitude

    @property
    def magnitude(self):
        return self.view(type=np.ndarray)

    @property
    def real(self):
        return Quantity(self.magnitude.real, self.dimensionality)

    @real.setter
    def real(self, r):
        self.magnitude.real = Quantity(r, self.dimensionality).magnitude

    @property
    def imag(self):
        return Quantity(self.magnitude.imag, self.dimensionality)

    @imag.setter
    def imag(self, i):
        self.magnitude.imag = Quantity(i, self.dimensionality).magnitude

    @property
    def simplified(self):
        rq = 1*unit_registry['dimensionless']
        for u, d in self.dimensionality.items():
            rq = rq * u.simplified**d
        return rq * self.magnitude

    @property
    def units(self):
        return Quantity(1.0, (self.dimensionality))
    @units.setter
    def units(self, units):
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

    def rescale(self, units=None):
        """
        Return a copy of the quantity converted to the specified units.
        If `units` is `None`, an attempt will be made to rescale the quantity
        to preferred units (see `rescale_preferred`).  
        """
        if units is None:
            try:
                return self.rescale_preferred()
            except Exception as e:
                raise Exception('No argument passed to `.rescale` and %s' % e)
        to_dims = validate_dimensionality(units)
        if self.dimensionality == to_dims:
            return self.astype(self.dtype)
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
    
    def rescale_preferred(self):
        """
        Return a copy of the quantity converted to the preferred units and scale.
        These will be identified from among the compatible units specified in the
        list PREFERRED in this module. For example, a voltage quantity might be
        converted to `mV`:
        ```
        import quantities as pq
        pq.quantity.PREFERRED = [pq.mV, pq.pA]
        old = 3.1415 * pq.V
        new = old.rescale_preferred() # `new` will be 3141.5 mV.
        ```
        """
        units_str = str(self.simplified.dimensionality)
        for preferred in PREFERRED:
            if units_str == str(preferred.simplified.dimensionality):
                return self.rescale(preferred)
        raise Exception(("Preferred units for '%s' (or equivalent) not specified in "
                        "quantites.quantity.PREFERRED." % self.dimensionality))
        
    @with_doc(np.ndarray.astype)
    def astype(self, dtype=None, **kwargs):
        '''Scalars are returned as scalar Quantity arrays.'''
        ret = super(Quantity, self.view(Quantity)).astype(dtype, **kwargs)
        # scalar quantities get converted to plain numbers, so we fix it
        # might be related to numpy ticket # 826
        if not isinstance(ret, type(self)):
            if self.__array_priority__ >= Quantity.__array_priority__:
                ret = type(self)(ret, self._dimensionality)
            else:
                ret = Quantity(ret, self._dimensionality)

        return ret

    def __array_finalize__(self, obj):
        self._dimensionality = getattr(obj, 'dimensionality', Dimensionality())

    def __array_prepare__(self, obj, context=None):
        if self.__array_priority__ >= Quantity.__array_priority__:
            res = obj if isinstance(obj, type(self)) else obj.view(type(self))
        else:
            # don't want a UnitQuantity
            res = obj.view(Quantity)
        if context is None:
            return res

        uf, objs, huh = context
        if uf.__name__.startswith('is'):
            return obj
        #print self, obj, res, uf, objs
        try:
            res._dimensionality = p_dict[uf](*objs)
        except KeyError:
            raise ValueError(
                """ufunc %r not supported by quantities
                please file a bug report at https://github.com/python-quantities
                """ % uf
                )
        return res

    def __array_wrap__(self, obj, context=None):
        if not isinstance(obj, Quantity):
            # backwards compatibility with numpy-1.3
            obj = self.__array_prepare__(obj, context)
        return obj

    @with_doc(np.ndarray.__add__)
    @scale_other_units
    def __add__(self, other):
        return super(Quantity, self).__add__(other)

    @with_doc(np.ndarray.__radd__)
    @scale_other_units
    def __radd__(self, other):
        return np.add(other, self)
        return super(Quantity, self).__radd__(other)

    @with_doc(np.ndarray.__iadd__)
    @scale_other_units
    def __iadd__(self, other):
        return super(Quantity, self).__iadd__(other)

    @with_doc(np.ndarray.__sub__)
    @scale_other_units
    def __sub__(self, other):
        return super(Quantity, self).__sub__(other)

    @with_doc(np.ndarray.__rsub__)
    @scale_other_units
    def __rsub__(self, other):
        return np.subtract(other, self)
        return super(Quantity, self).__rsub__(other)

    @with_doc(np.ndarray.__isub__)
    @scale_other_units
    def __isub__(self, other):
        return super(Quantity, self).__isub__(other)

    @with_doc(np.ndarray.__mod__)
    @scale_other_units
    def __mod__(self, other):
        return super(Quantity, self).__mod__(other)

    @with_doc(np.ndarray.__imod__)
    @scale_other_units
    def __imod__(self, other):
        return super(Quantity, self).__imod__(other)

    @with_doc(np.ndarray.__imul__)
    @protected_multiplication
    def __imul__(self, other):
        return super(Quantity, self).__imul__(other)

    @with_doc(np.ndarray.__rmul__)
    def __rmul__(self, other):
        return np.multiply(other, self)
        return super(Quantity, self).__rmul__(other)

    @with_doc(np.ndarray.__itruediv__)
    @protected_multiplication
    def __itruediv__(self, other):
        return super(Quantity, self).__itruediv__(other)

    @with_doc(np.ndarray.__rtruediv__)
    def __rtruediv__(self, other):
        return np.true_divide(other, self)
        return super(Quantity, self).__rtruediv__(other)

    if sys.version_info[0] < 3:
        @with_doc(np.ndarray.__idiv__)
        @protected_multiplication
        def __idiv__(self, other):
            return super(Quantity, self).__itruediv__(other)

        @with_doc(np.ndarray.__rdiv__)
        def __rdiv__(self, other):
            return np.divide(other, self)

    @with_doc(np.ndarray.__pow__)
    @check_uniform
    def __pow__(self, other):
        return np.power(self, other)

    @with_doc(np.ndarray.__ipow__)
    @check_uniform
    @protected_power
    def __ipow__(self, other):
        return super(Quantity, self).__ipow__(other)

    def __round__(self, decimals=0):
        return np.around(self, decimals)

    @with_doc(np.ndarray.__repr__)
    def __repr__(self):
        return '%s * %s'%(
            repr(self.magnitude), self.dimensionality.string
        )

    @with_doc(np.ndarray.__str__)
    def __str__(self):
        if markup.config.use_unicode:
            dims = self.dimensionality.unicode
        else:
            dims = self.dimensionality.string
        return '%s %s'%(str(self.magnitude), dims)

    if tuple(map(int, np.__version__.split('.')[:2])) >= (1, 14):
        # in numpy 1.14 the formatting of scalar values was changed
        # see https://github.com/numpy/numpy/pull/9883

        def __format__(self, format_spec):
            ret = super(Quantity, self).__format__(format_spec)
            if self.ndim:
                return ret
            return ret + ' {}'.format(self.dimensionality)

    @with_doc(np.ndarray.__getitem__)
    def __getitem__(self, key):
        ret = super(Quantity, self).__getitem__(key)
        if isinstance(ret, Quantity):
            return ret
        else:
            return Quantity(ret, self._dimensionality)

    @with_doc(np.ndarray.__setitem__)
    def __setitem__(self, key, value):
        if not isinstance(value, Quantity):
            value = Quantity(value)
        if self._dimensionality != value._dimensionality:
            value = value.rescale(self._dimensionality)
        self.magnitude[key] = value

    @with_doc(np.ndarray.__lt__)
    @wrap_comparison
    def __lt__(self, other):
        return self.magnitude < other

    @with_doc(np.ndarray.__le__)
    @wrap_comparison
    def __le__(self, other):
        return self.magnitude <= other

    @with_doc(np.ndarray.__eq__)
    def __eq__(self, other):
        if isinstance(other, Quantity):
            try:
                other = other.rescale(self._dimensionality).magnitude
            except ValueError:
                return np.zeros(self.shape, '?')
        return self.magnitude == other

    @with_doc(np.ndarray.__ne__)
    def __ne__(self, other):
        if isinstance(other, Quantity):
            try:
                other = other.rescale(self._dimensionality).magnitude
            except ValueError:
                return np.ones(self.shape, '?')
        return self.magnitude != other

    @with_doc(np.ndarray.__ge__)
    @wrap_comparison
    def __ge__(self, other):
        return self.magnitude >= other

    @with_doc(np.ndarray.__gt__)
    @wrap_comparison
    def __gt__(self, other):
        return self.magnitude > other

    #I don't think this implementation is particularly efficient,
    #perhaps there is something better
    @with_doc(np.ndarray.tolist)
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

    @with_doc(np.ndarray.sum)
    def sum(self, axis=None, dtype=None, out=None):
        ret = self.magnitude.sum(axis, dtype, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.nansum)
    def nansum(self, axis=None, dtype=None, out=None):
        import numpy as np
        return Quantity(
            np.nansum(self.magnitude, axis, dtype, out),
            self.dimensionality,
            copy=False
        )

    @with_doc(np.ndarray.fill)
    def fill(self, value):
        self.magnitude.fill(value)
        try:
            self._dimensionality = value.dimensionality
        except AttributeError:
            pass

    @with_doc(np.ndarray.put)
    def put(self, indicies, values, mode='raise'):
        """
        performs the equivalent of ndarray.put() but enforces units
        values - must be an Quantity with the same units as self
        """
        if not isinstance(values, Quantity):
            values = Quantity(values)
        if values._dimensionality != self._dimensionality:
            values = values.rescale(self.units)
        self.magnitude.put(indicies, values, mode)

    # choose does not function correctly, and it is not clear
    # how it would function, so for now it will not be implemented

    @with_doc(np.ndarray.argsort)
    def argsort(self, axis=-1, kind='quick', order=None):
        return self.magnitude.argsort(axis, kind, order)

    @with_doc(np.ndarray.searchsorted)
    def searchsorted(self,values, side='left'):
        if not isinstance (values, Quantity):
            values = Quantity(values, copy=False)

        if values._dimensionality != self._dimensionality:
            raise ValueError("values does not have the same units as self")

        return self.magnitude.searchsorted(values.magnitude, side)

    @with_doc(np.ndarray.nonzero)
    def nonzero(self):
        return self.magnitude.nonzero()

    @with_doc(np.ndarray.max)
    def max(self, axis=None, out=None):
        ret = self.magnitude.max(axis, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.argmax)
    def argmax(self, axis=None, out=None):
        return self.magnitude.argmax(axis, out)

    @with_doc(np.nanmax)
    def nanmax(self, axis=None, out=None):
        return Quantity(
            np.nanmax(self.magnitude),
            self.dimensionality,
            copy=False
        )

    @with_doc(np.ndarray.min)
    def min(self, axis=None, out=None):
        ret = self.magnitude.min(axis, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.nanmin)
    def nanmin(self, axis=None, out=None):
        return Quantity(
            np.nanmin(self.magnitude),
            self.dimensionality,
            copy=False
        )

    @with_doc(np.ndarray.argmin)
    def argmin(self, axis=None, out=None):
        return self.magnitude.argmin(axis, out)

    @with_doc(np.nanargmin)
    def nanargmin(self,axis=None, out=None):
        return np.nanargmin(self.magnitude)

    @with_doc(np.nanargmax)
    def nanargmax(self,axis=None, out=None):
        return np.nanargmax(self.magnitude)

    @with_doc(np.ndarray.ptp)
    def ptp(self, axis=None, out=None):
        ret = self.magnitude.ptp(axis, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.clip)
    def clip(self, min=None, max=None, out=None):
        if min is None and max is None:
            raise ValueError("at least one of min or max must be set")
        else:
            if min is None: min = Quantity(-np.Inf, self._dimensionality)
            if max is None: max = Quantity(np.Inf, self._dimensionality)

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
        dim = self.dimensionality
        if out is None:
            return Quantity(clipped, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.round)
    def round(self, decimals=0, out=None):
        ret = self.magnitude.round(decimals, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.trace)
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        ret = self.magnitude.trace(offset, axis1, axis2, dtype, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.squeeze)
    def squeeze(self, axis=None):
        return Quantity(
            self.magnitude.squeeze(axis),
            self.dimensionality,
            copy=False
        )

    @with_doc(np.ndarray.mean)
    def mean(self, axis=None, dtype=None, out=None):
        ret = self.magnitude.mean(axis, dtype, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.nanmean)
    def nanmean(self, axis=None, dtype=None, out=None):
        import numpy as np
        return Quantity(
            np.nanmean(self.magnitude, axis, dtype, out),
            self.dimensionality,
            copy=False)

    @with_doc(np.ndarray.var)
    def var(self, axis=None, dtype=None, out=None, ddof=0):
        ret = self.magnitude.var(axis, dtype, out, ddof)
        dim = self._dimensionality**2
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.std)
    def std(self, axis=None, dtype=None, out=None, ddof=0):
        ret = self.magnitude.std(axis, dtype, out, ddof)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.nanstd)
    def nanstd(self, axis=None, dtype=None, out=None, ddof=0):
        return Quantity(
            np.nanstd(self.magnitude, axis, dtype, out, ddof),
            self._dimensionality,
            copy=False
        )

    @with_doc(np.ndarray.prod)
    def prod(self, axis=None, dtype=None, out=None):
        if axis == None:
            power = self.size
        else:
            power = self.shape[axis]

        ret = self.magnitude.prod(axis, dtype, None if out is None else out.magnitude)
        dim = self._dimensionality**power
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.cumsum)
    def cumsum(self, axis=None, dtype=None, out=None):
        ret = self.magnitude.cumsum(axis, dtype, None if out is None else out.magnitude)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if not isinstance(out, Quantity):
            raise TypeError("out parameter must be a Quantity")
        out._dimensionality = dim
        return out

    @with_doc(np.ndarray.cumprod)
    def cumprod(self, axis=None, dtype=None, out=None):
        if self._dimensionality:
            # different array elements would have different dimensionality
            raise ValueError(
                "Quantity must be dimensionless, try using simplified"
            )

        ret = self.magnitude.cumprod(axis, dtype, out)
        dim = self.dimensionality
        if out is None:
            return Quantity(ret, dim, copy=False)
        if isinstance(out, Quantity):
            out._dimensionality = dim
        return out

    # list of unsupported functions: [choose]

    def __setstate__(self, state):
        ndarray_state = state[:-1]
        units = state[-1]
        np.ndarray.__setstate__(self, ndarray_state)
        self._dimensionality = units

    def __reduce__(self):
        """
        Return a tuple for pickling a Quantity.
        """
        reconstruct,reconstruct_args,state = super(Quantity,self).__reduce__()
        state = state + (self._dimensionality,)
        return (_reconstruct_quantity,
                (self.__class__, np.ndarray, (0, ), 'b', ),
                state)

    def __deepcopy__(self, memo_dict):
        # constructor copies by default
        return Quantity(self.magnitude, self.dimensionality)


def _reconstruct_quantity(subtype, baseclass, baseshape, basetype,):
    """Internal function that builds a new MaskedArray from the
    information stored in a pickle.

    """
    _data = np.ndarray.__new__(baseclass, baseshape, basetype)
    return subtype.__new__(subtype, _data, dtype=basetype,)
