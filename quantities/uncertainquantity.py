# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

import sys

import numpy as np

from . import markup
from .quantity import Quantity, scale_other_units
from .registry import unit_registry
from .decorators import with_doc

class UncertainQuantity(Quantity):

    # TODO: what is an appropriate value?
    __array_priority__ = 22

    def __new__(cls, data, units='', uncertainty=None, dtype='d', copy=True):
        ret = Quantity.__new__(cls, data, units, dtype, copy)
        # _uncertainty initialized to be dimensionless by __array_finalize__:
        ret._uncertainty._dimensionality = ret._dimensionality

        if uncertainty is not None:
            ret.uncertainty = Quantity(uncertainty, ret._dimensionality)
        elif isinstance(data, UncertainQuantity):
            if copy or ret._dimensionality != uncertainty._dimensionality:
                uncertainty = data.uncertainty.rescale(ret.units)
            ret.uncertainty = uncertainty

        return ret

    @Quantity.units.setter
    def units(self, units):
        super(UncertainQuantity, self)._set_units(units)
        self.uncertainty.units = self._dimensionality

    @property
    def _reference(self):
        ret = super(UncertainQuantity, self)._reference.view(UncertainQuantity)
        ret.uncertainty = self.uncertainty._reference
        return ret

    @property
    def simplified(self):
        ret = super(UncertainQuantity, self).simplified.view(UncertainQuantity)
        ret.uncertainty = self.uncertainty.simplified
        return ret

    @property
    def uncertainty(self):
        return self._uncertainty
    @uncertainty.setter
    def uncertainty(self, uncertainty):
        if not isinstance(uncertainty, Quantity):
            uncertainty = Quantity(uncertainty, copy=False)
        try:
            assert self.shape == uncertainty.shape
        except AssertionError:
            raise ValueError('data and uncertainty must have identical shape')
        if uncertainty._dimensionality != self._dimensionality:
            uncertainty = uncertainty.rescale(self._dimensionality)
        self._uncertainty = uncertainty

    @property
    def relative_uncertainty(self):
        return self.uncertainty.magnitude/self.magnitude

    @with_doc(Quantity.rescale, use_header=False)
    def rescale(self, units):
        cls = UncertainQuantity
        ret = super(cls, self).rescale(units).view(cls)
        ret.uncertainty = self.uncertainty.rescale(units)
        return ret

    def __array_finalize__(self, obj):
        Quantity.__array_finalize__(self, obj)
        self._uncertainty = getattr(
            obj,
            'uncertainty',
            Quantity(
                np.zeros(self.shape, self.dtype),
                self._dimensionality,
                copy=False
            )
        )

    @with_doc(Quantity.__add__, use_header=False)
    @scale_other_units
    def __add__(self, other):
        res = super(UncertainQuantity, self).__add__(other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        return UncertainQuantity(res, uncertainty=u, copy=False)

    @with_doc(Quantity.__radd__, use_header=False)
    @scale_other_units
    def __radd__(self, other):
        return self.__add__(other)

    @with_doc(Quantity.__sub__, use_header=False)
    @scale_other_units
    def __sub__(self, other):
        res = super(UncertainQuantity, self).__sub__(other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        return UncertainQuantity(res, uncertainty=u, copy=False)

    @with_doc(Quantity.__rsub__, use_header=False)
    @scale_other_units
    def __rsub__(self, other):
        if not isinstance(other, UncertainQuantity):
            other = UncertainQuantity(other, copy=False)

        return UncertainQuantity.__sub__(other, self)

    @with_doc(Quantity.__mul__, use_header=False)
    def __mul__(self, other):
        res = super(UncertainQuantity, self).__mul__(other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            u = res.view(Quantity) * ru
        except AttributeError:
            other = np.array(other, copy=False, subok=True)
            u = (self.uncertainty**2*other**2)**0.5

        res._uncertainty = u
        return res

    @with_doc(Quantity.__rmul__, use_header=False)
    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self*-1

    @with_doc(Quantity.__truediv__, use_header=False)
    def __truediv__(self, other):
        res = super(UncertainQuantity, self).__truediv__(other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            u = res.view(Quantity) * ru
        except AttributeError:
            other = np.array(other, copy=False, subok=True)
            u = (self.uncertainty**2/other**2)**0.5

        res._uncertainty = u
        return res

    @with_doc(Quantity.__rtruediv__, use_header=False)
    def __rtruediv__(self, other):
        temp = UncertainQuantity(
            1/self.magnitude, self.dimensionality**-1,
            self.relative_uncertainty/self.magnitude, copy=False
        )
        return other * temp

    if sys.version_info[0] < 3:
        __div__ = __truediv__
        __rdiv__ = __rtruediv__

    @with_doc(Quantity.__pow__, use_header=False)
    def __pow__(self, other):
        res = super(UncertainQuantity, self).__pow__(other)
        res.uncertainty = res.view(Quantity) * other * self.relative_uncertainty
        return res

    @with_doc(Quantity.__getitem__, use_header=False)
    def __getitem__(self, key):
        return UncertainQuantity(
            self.magnitude[key],
            self._dimensionality,
            self.uncertainty[key],
            copy=False
        )

    @with_doc(Quantity.__repr__, use_header=False)
    def __repr__(self):
        return '%s(%s, %s, %s)'%(
            self.__class__.__name__,
            repr(self.magnitude),
            self.dimensionality.string,
            repr(self.uncertainty.magnitude)
        )

    @with_doc(Quantity.__str__, use_header=False)
    def __str__(self):
        if markup.config.use_unicode:
            dims = self.dimensionality.unicode
        else:
            dims = self.dimensionality.string
        s = '%s %s\n+/-%s (1 sigma)'%(
            str(self.magnitude),
            dims,
            str(self.uncertainty)
        )
        if markup.config.use_unicode:
            return s.replace('+/-', '±').replace(' sigma', 'σ')
        return s

    @with_doc(np.ndarray.sum)
    def sum(self, axis=None, dtype=None, out=None):
        return UncertainQuantity(
            self.magnitude.sum(axis, dtype, out),
            self.dimensionality,
            (np.sum(self.uncertainty.magnitude**2, axis))**0.5,
            copy=False
        )

    @with_doc(np.nansum)
    def nansum(self, axis=None, dtype=None, out=None):
        return UncertainQuantity(
            np.nansum(self.magnitude, axis, dtype, out),
            self.dimensionality,
            (np.nansum(self.uncertainty.magnitude**2, axis))**0.5,
            copy=False
        )

    @with_doc(np.ndarray.mean)
    def mean(self, axis=None, dtype=None, out=None):
        return UncertainQuantity(
            self.magnitude.mean(axis, dtype, out),
            self.dimensionality,
            ((1.0/np.size(self,axis))**2 * np.sum(self.uncertainty.magnitude**2, axis))**0.5,
            copy=False)

    @with_doc(np.nanmean)
    def nanmean(self, axis=None, dtype=None, out=None):
        size = np.sum(~np.isnan(self),axis)
        return UncertainQuantity(
            np.nanmean(self.magnitude, axis, dtype, out),
            self.dimensionality,
            ((1.0/size)**2 * np.nansum(np.nan_to_num(self.uncertainty.magnitude)**2, axis))**0.5,
            copy=False)

    @with_doc(np.sqrt)
    def sqrt(self, out=None):
        return self**0.5

    @with_doc(np.ndarray.max)
    def max(self, axis=None, out=None):
        idx = np.unravel_index(np.argmax(self.magnitude), self.shape)
        return self[idx]

    @with_doc(np.nanmax)
    def nanmax(self, axis=None, out=None):
        idx = np.unravel_index(np.nanargmax(self.magnitude), self.shape)
        return self[idx]

    @with_doc(np.ndarray.min)
    def min(self, axis=None, out=None):
        idx = np.unravel_index(np.argmin(self.magnitude), self.shape)
        return self[idx]

    @with_doc(np.nanmin)
    def nanmin(self, axis=None, out=None):
        idx = np.unravel_index(np.nanargmin(self.magnitude), self.shape)
        return self[idx]

    @with_doc(np.ndarray.argmin)
    def argmin(self,axis=None, out=None):
        return self.magnitude.argmin()

    @with_doc(np.ndarray.argmax)
    def argmax(self,axis=None, out=None):
        return self.magnitude.argmax()

    @with_doc(np.nanargmin)
    def nanargmin(self,axis=None, out=None):
        return np.nanargmin(self.magnitude)

    @with_doc(np.nanargmax)
    def nanargmax(self,axis=None, out=None):
        return np.nanargmax(self.magnitude)

    def __setstate__(self, state):
        ndarray_state = state[:-2]
        units, sigma = state[-2:]
        np.ndarray.__setstate__(self, ndarray_state)
        self._dimensionality = units
        self._uncertainty = sigma

    def __reduce__(self):
        """
        Return a tuple for pickling a Quantity.
        """
        reconstruct, reconstruct_args, state = super(UncertainQuantity, self).__reduce__()
        state = state + (self._uncertainty,)
        return reconstruct, reconstruct_args, state

