"""
"""

import numpy

from quantities.quantity import Quantity
from quantities.registry import unit_registry

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
            raise ValueError('data and uncertainty must have identical shape')
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
        return UncertainQuantity(res, uncertainty=u, copy=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res = Quantity.__sub__(self, other)
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
            other = numpy.array(other, copy=False)
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
        return '%s %s\n+/-%s (1 sigma)'%(
            numpy.ndarray.__str__(self),
            str(self.dimensionality),
            self.uncertainty
        )

    __str__ = __repr__
