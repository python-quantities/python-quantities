"""
"""

import copy

from numpy import array, ndarray

from dimensionality import Dimensionality
from parser import unit_registry


# TODO: this should get moved into units, which should be a Quantity subclass
class ProtectedUnitsError(Exception):

    def __init__(self, units):
        self._units = units
        return

    def __str__(self):
        str = "reference unit %s should not be modified" % (self._units)
        return str


class Quantity(ndarray):

    """
    """

    __array_priority__ = 21

    def __new__(cls, data, units='', dtype=None, copy=True, static=False):
        if isinstance(units, str):
            units = unit_registry[units]
        if isinstance(units, Quantity):
            scaling = float(units)
            offset = 0.0
            units = units.units
        elif isinstance(units, Dimensionality):
            # TODO need to get rid of this possibility by creating unit subclass
            scaling, offset = units.scaling
        else:
            raise "units must be a string or a valid combination of units"
#        if isinstance(data, Quantity):
#            dtype2 = data.dtype
#            if (dtype is None):
#                dtype = dtype2
#            if (dtype2 == dtype) and (not copy):
#                data._units = units
#                return data
#            new = data.astype(dtype)
#            new._units = units
#            return new

        if not isinstance(data, ndarray):
            data = array(data, dtype=dtype, copy=copy)
            shape = data.shape

            if not data.flags.contiguous:
                data = data.copy()

#        scaling = units.scaling
        if not scaling == 1: data *= scaling
        if offset: data += offset
        ret = ndarray.__new__(cls, data.shape, data.dtype, buffer=data)
        ret._units = units
        ret._static = static
        return ret

    def __array_finalize__(self, obj):
        self._units = copy.deepcopy(getattr(obj, '_units', None))
        self._static = False

    def __deepcopy__(self, memo={}):
        units = copy.deepcopy(self.units)
        return self.__class__(self.view(type=ndarray), units)

    def _get_units(self):
        return self._units
    def _set_units(self, units):
        if self._static: raise ProtectedUnitsError(str(units))
        self._units.set_units(units)
        scaling, offset = self._units.scaling
        if not scaling == 1.0:
            super(Quantity, self).__imul__(scaling)
        if offset:
            super(Quantity, self).__iadd__(offset)
    units = property(_get_units, _set_units)

    def _get_static(self):
        return self._static
    def _set_static(self, val):
        self._static = bool(val)
    static = property(_get_static, _set_static)

    def modify_units(self, units):
        if self._static: raise ProtectedUnitsError(str(units))
        self._units.set_units(units, strict=False)
        scaling, offset = self._units.scaling
        if not scaling == 1.0:
            super(Quantity, self).__imul__(scaling)
        if offset:
            super(Quantity, self).__iadd__(offset)

    def simplify_units(self):
        if self._static: raise ProtectedUnitsError('')
        q = self._units.simplify_units()
        s, o = self._units.scaling
        assert s == 1
        assert o == 0
        self._units = q.units
        super(Quantity, self).__imul__(float(q))

    def __str__(self):
        return super(Quantity, self).__str__() + ' ' + str(self.units)

    def __repr__(self):
        return super(Quantity, self).__repr__() + ', ' + str(self.units)

    def __add__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, '')
        units = self._units + other._units
        scaling, offset = units.scaling
        other = other.view(type=ndarray)
        if not scaling == 1.0:
            other *= scaling
        if offset:
            other += offset
        ret = super(Quantity, self).__add__(other)
        ret._units = units
        return ret

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            other = Quantity(other, '')
        units = self._units - other._units
        scaling, offset = units.scaling
        other = other.view(type=ndarray)
        if not scaling == 1.0:
            other *= scaling
        if offset:
            other += offset
        ret = super(Quantity, self).__sub__(other)
        ret._units = units
        return ret

    def __mul__(self, other):
        if not isinstance(other, Quantity):
            return super(Quantity, self).__mul__(other)
        units = self._units * other._units
        scaling, offset = units.scaling
        other = other.view(type=ndarray)
        if not scaling == 1.0:
            other *= scaling
        if offset:
            other += offset
        ret = super(Quantity, self).__mul__(other)
        ret._units = units
        return ret

    def __div__(self, other):
        if not isinstance(other, Quantity):
            return super(Quantity, self).__div__(other)
        units = self._units / other._units
        scaling, offset = units.scaling
        other = other.view(type=ndarray)
        if not scaling == 1.0:
            other *= scaling
        if offset:
            other += offset
        ret = super(Quantity, self).__div__(other)
        ret._units = units
        return ret

    def __rdiv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        other = float(other)
        units = self._units**other
        ret = super(Quantity, self).__pow__(other)
        ret._units = units
        return ret

#    def __radd__(self,other):
#        if not isinstance(other, (Quantity,int,long,float)):
#            raise "Error must add a number or a udunit object"
#        out=Quantity(self.units,self.value)
#        if isinstance(other, Quantity):
#            s,i=_udunits.convert(other.units,self.units)
#            out.value=self.value+other.value*s+i
#        else:
#            out.value+=other
#        return out
#
#    def __rsub__(self,other):
#        if not isinstance(other, (Quantity,int,long,float)):
#            raise "Error must sub a number or a udunit object"
#        out=Quantity(self.units,self.value)
#        if isinstance(other, Quantity):
#            s,i=_udunits.convert(other.units,self.units)
#            out.value=(other.value*s+i)-self.value
#        else:
#            out.value=other-out.value
#        return out
#
#    def __rmul__(self,other):
#        if not isinstance(other, (Quantity,int,long,float)):
#            raise "Error must multiply a number or a udunit object"
#        out=Quantity(self.units+'*'+self.units,self.value)
#        if isinstance(other, Quantity):
#            try:
#                s,i=_udunits.convert(other.units,self.units)
#                out.value=self.value*(other.value*s+i)
#            except: #Ok uncompatible units, just do the produce
#                out=Quantity(self.units+'*'+other.units,self.value*other.value)
#        else:
#            out = Quantity(other*self.value,self.units)
#        return out
#
#    def __rdiv__(self,other):
#        if not isinstance(other, (Quantity,int,long,float)):
#            raise "Error must divide by a number or a udunit object"
#        out=Quantity(self.units,self.value)
#        if isinstance(other, units):
#            try:
#                s,i=_udunits.convert(other.units,self.units)
#                out=(other.value*s+i)/self.value
#            except: #Ok uncompatible units, just do the produce
#                out=Quantity(other.value/self.value,other.units+'/'+self.units)
#        else:
#            out.value=other/self.value
#            out._units='1/('+self.units+')'
#        return out
