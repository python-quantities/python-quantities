# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

import numpy

from .config import USE_UNICODE
from .quantity import Quantity
from .registry import unit_registry

class UncertainQuantity(Quantity):

    # TODO: what is an appropriate value?
    __array_priority__ = 22

    def __new__(cls, data, units='', uncertainty=None, dtype='d', copy=True):
        ret = Quantity.__new__(cls, data, units, dtype, copy)

        if uncertainty is None:
            if isinstance(data, UncertainQuantity):
                if copy:
                    uncertainty = data.uncertainty.copy()
                else:
                    uncertainty = data.uncertainty
            else:
                uncertainty = numpy.zeros(ret.shape, dtype)
        elif not isinstance(uncertainty, numpy.ndarray):
            uncertainty = numpy.array(uncertainty, dtype)
        try:
            assert uncertainty.shape == ret.shape
        except AssertionError:
            raise ValueError('data and uncertainty must have identical shape')
        ret.uncertainty = uncertainty

        return ret

    def _set_units(self, units):
        super(UncertainQuantity, self)._set_units(units)
        self.uncertainty.units = self.units
    units = property(Quantity._get_units, _set_units)

    @property
    def simplified(self):
        ret = super(UncertainQuantity, self).simplified.view(UncertainQuantity)
        ret.uncertainty = self.uncertainty.simplified
        return ret

    def get_uncertainty(self):
        return self._uncertainty
    def set_uncertainty(self, uncertainty):
        if not isinstance(uncertainty, Quantity):
            uncertainty = Quantity(uncertainty, self.units, copy=False)
        try:
            assert self.shape == uncertainty.shape
        except AssertionError:
            raise ValueError('data and uncertainty must have identical shape')
        if uncertainty.units != self.units:
            uncertainty = uncertainty.rescale(self.units)
        self._uncertainty = uncertainty
    uncertainty = property(get_uncertainty, set_uncertainty)

    @property
    def relative_uncertainty(self):
        return self.uncertainty.magnitude/self.magnitude

    def rescale(self, units):
        """
        Return a copy of the quantity converted to the specified units
        """
        cls = UncertainQuantity
        ret = super(cls, self).rescale(units).view(cls)
        ret.uncertainty = self.uncertainty.rescale(units)
        return ret

    def __array_finalize__(self, obj):
        Quantity.__array_finalize__(self, obj)
        self._uncertainty = getattr(
            obj,
            'uncertainty',
            Quantity(numpy.zeros(self.shape, self.dtype), self.units, copy=False)
        )

    def __add__(self, other):
        res = super(UncertainQuantity, self).__add__(other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res = super(UncertainQuantity, self).__sub__(other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __rsub__(self, other):
        if not isinstance(other, UncertainQuantity):
            other = UncertainQuantity(other, copy=False)

        return UncertainQuantity.__sub__(other, self)

    def __mul__(self, other):
        res = super(UncertainQuantity, self).__mul__(other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            u = res.view(Quantity) * ru
        except AttributeError:
            other = numpy.array(other, copy=False, subok=True)
            u = (self.uncertainty**2*other**2)**0.5

        res._uncertainty = u
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        res = super(UncertainQuantity, self).__truediv__(other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            u = res.view(Quantity) * ru
        except AttributeError:
            other = numpy.array(other, copy=False)
            u = (self.uncertainty**2/other**2)**0.5

        res._uncertainty = u
        return res

    def __rtruediv__(self, other):
        temp = UncertainQuantity(
            1/self.magnitude, self.dimensionality**-1,
            1/self.uncertainty.magnitude, copy=False
        )
        return other * temp

    def __pow__(self, other):
        res = super(UncertainQuantity, self).__pow__(other)
        res.uncertainty = res.view(Quantity) * other * self.relative_uncertainty
        return res

    def __getitem__(self, key):
        return UncertainQuantity(
            self.magnitude[key],
            self.units,
            self.uncertainty[key],
            copy=False
        )

    def __repr__(self):
        return '%s(%s, %s, %s)'%(
            self.__class__.__name__,
            repr(self.magnitude),
            repr(self.dimensionality),
            repr(self.uncertainty.magnitude)
        )

    def __str__(self):
        s = '%s %s\n+/-%s (1 sigma)'%(
            str(self.magnitude),
            str(self.dimensionality),
            str(self.uncertainty)
        )
        if USE_UNICODE:
            return s.replace('+/-', '±').replace(' sigma', 'σ')
        return s
