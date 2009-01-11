"""
"""

import numpy

from quantities.quantity import Quantity
from quantities.registry import unit_registry

class UncertainQuantity(Quantity):

    # TODO: what is an appropriate value?
    __array_priority__ = 22

    def __new__(cls, data, units='', uncertainty=None, dtype='d', copy=True):
        return Quantity.__new__(cls, data, units, dtype, copy)

    def __init__(self, data, units='', uncertainty=None, dtype='d', copy=True):
        Quantity.__init__(self, data, units, dtype, copy)

        if uncertainty is None:
            if isinstance(data, UncertainQuantity):
                uncertainty = data.uncertainty
            else:
                uncertainty = numpy.zeros(self.shape, dtype)
        elif not isinstance(uncertainty, numpy.ndarray):
            uncertainty = numpy.array(uncertainty, dtype)
        try:
            assert uncertainty.shape == self.shape
        except AssertionError:
            raise ValueError('data and uncertainty must have identical shape')
        self.uncertainty = uncertainty

    @property
    def simplified(self):
        sq = unit_registry['dimensionless']
        for u, d in self.dimensionality.iteritems():
            sq = sq * u.reference_quantity**d
        u = self.uncertainty.simplified
        # TODO: use view:
        return UncertainQuantity(sq * self.magnitude, uncertainty=u, copy=False)

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
            assert self.shape == uncertainty.shape
            uncertainty.units = self.units
            self._uncertainty = uncertainty
        except AssertionError:
            ValueError('data and uncertainty must have identical shape')
    uncertainty = property(get_uncertainty, set_uncertainty)

    @property
    def relative_uncertainty(self):
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
            obj,
            'uncertainty',
            Quantity(numpy.zeros(self.shape, self.dtype), self.units)
        )

    def __add__(self, other):
        res = Quantity.__add__(self, other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __sub__(self, other):
        res = Quantity.__sub__(self, other)
        u = (self.uncertainty**2+other.uncertainty**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __mul__(self, other):
        res = Quantity.__mul__(self, other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            u = res * ru
        except AttributeError:
            other = numpy.array(other, copy=False)
            u = (self.uncertainty**2*other**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __truediv__(self, other):
        res = Quantity.__truediv__(self, other)
        try:
            sru = self.relative_uncertainty
            oru = other.relative_uncertainty
            ru = (sru**2+oru**2)**0.5
            u = res * ru
        except AttributeError:
            other = numpy.array(other, copy=False)
            u = (self.uncertainty**2/other**2)**0.5
        # TODO: use .view:
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __pow__(self, other):
        res = Quantity.__pow__(self, other)
        ru = other * self.relative_uncertainty
        u = res * ru
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __getitem__(self, key):
        return UncertainQuantity(
            self.magnitude[key],
            self.units,
            self.uncertainty[key],
            copy=False
        )

    def __repr__(self):
        return '%s*%s\n+/-%s (1 sigma)'%(
            numpy.ndarray.__str__(self),
            str(self.dimensionality),
            self.uncertainty
        )

    __str__ = __repr__
