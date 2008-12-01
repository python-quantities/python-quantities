"""
"""

import copy
import os
import warnings

from numpy import allclose, array, ndarray

import udunits as _udunits

_udunits.init(os.path.join(os.path.dirname(__file__),
                           'quantities-data',
                           'udunits.dat'))

from parser import unit_registry


class IncompatibleUnits(Exception):

    def __init__(self, op, operand1, operand2):
        self._op = op
        self._op1 = operand1
        self._op2 = operand2
        return

    def __str__(self):
        str = "Cannot %s quanitites with units of '%s' and '%s'" % \
              (self._op, self._op1, self._op2)
        return str


class Dimensionality(object):

    """
    """

    def __init__(self, units, dimensions, *args, **kwargs):
        # mass, length, time, current,luminous_intensity, substance
        # temperature, information, angle, currency, compound
        assert len(units) == len(dimensions)
        assert isinstance(dimensions, ndarray)
        self._dimensions = dimensions
        self._units = units

        self._compound = {}
        for arg in args:
            assert isinstance(arg, str)
            if arg in self._compound: self._compound[arg] += 1
            else: self._compound[arg] = 1
        for k, v in kwargs.items():
            if k in self._compound: self._compound[k] += v
            else: self._compound[k] = v

        self._scaling = 1.0
        self._offset = 0.0

    def __deepcopy__(self, memo={}):
        compound = copy.deepcopy(self._compound)
        units = copy.deepcopy(self._units)
        dimensions = copy.deepcopy(self._dimensions)
        return self.__class__(units, dimensions, **compound)

    def simplify_units(self):
        scaling = unit_registry['dimensionless']
        for k, v in self._compound.items():
            q = unit_registry[k]**v
            scaling = scaling * q
        return scaling

    def get_units(self):
        units = list(copy.copy(self._units))
        for k, v in self._compound.items():
            units.append(k)
        return units
    def set_units(self, units, strict=True):
        if isinstance(units, str):
            units = unit_registry[units]
        if isinstance(units.units, Dimensionality):
            # TODO: This should be improved
            scaling = float(units)
            offset = 0.0
            units = copy.deepcopy(units.units)
        elif isinstance(units, Dimensionality):
            scaling, offset = units.scaling
            assert scaling == 1.0
            assert offset == 0
        else:
            raise "units must be a string or a valid combination of units"

        if strict or len(units._compound):
            if not strict: print "warning, compound unit conversions must match dimensionality"
            try:
                scaling, offset = _udunits.convert(self._udunits(), units._udunits())
                self._scaling *= scaling
                self._offset += offset
                self._units = copy.deepcopy(units._units)
                self._dimensions = copy.deepcopy(units._dimensions)
                self._compound = copy.deepcopy(units._compound)
            except TypeError:
                raise IncompatibleUnits('convert between', self._udunits(), units._udunits())
        else:
            new_units = copy.copy(self._units)
            old = []
            new = []
            for i, (su, ou) in enumerate(zip(self._units, units._units)):
                if ou and not su:
                    new_units[i] = ou
                if (su and ou) and not (su == ou):
                    old.append(su+'^%s'%self._dimensions[i])
                    new.append(ou+'^%s'%self._dimensions[i])
                    units[i] = ou
            old = '*'.join(old)
            new = '*'.join(new)
            scaling, offset = _udunits.convert(old, new)
            self._scaling = scaling
            self._offset = offset
            self._units = new_units
    units = property(get_units, set_units)

    @property
    def dimensions(self):
        dimensions = list(copy.copy(self._dimensions))
        for k, v in self._compound.items():
            dimensions.append(v)
        return array(dimensions)

    @property
    def scaling(self):
        s = self._scaling
        o = self._offset
        self._scaling = 1.0
        self._offset = 0.0
        return s, o

    def __repr__(self):
        num = []
        den = []
        for u, d in zip(self.units, self.dimensions):
            if d>0:
                if d != 1: u = u + ('^%s'%d).rstrip('.0')
                num.append(u)
            elif d<0:
                d = -d
                if d != 1: u = u + ('^%s'%d).rstrip('.0')
                den.append(u)
        res = ' * '.join(num)
        if len(den):
            if not res: res = '1'
            res = res + ' / ' + ' '.join(den)
        if not res: res = '(dimensionless)'
        return res

    def _udunits(self):
        if not allclose(array(self.dimensions, 'i'), self.dimensions):
            warnings.warn('udunits will not convert units with fractional exponents')
        num = []
        den = []
        for u, d in zip(self.units, self.dimensions):
            if d>0:
                if d != 1: u = u + ('^%s'%d).rstrip('.0')
                num.append(u)
            elif d<0:
                if d != 1: u = u + ('^%s'%d).rstrip('.0')
                den.append(u)
        res = ' '.join(num)
        if len(den):
            if not res: res = '1'
            res = res + ' ' + ' '.join(den)
        res = res.replace('(', '').replace(')', '').replace('**', '^')
        return res

    def __add__(self, other):
        assert isinstance(other, Dimensionality)
        if not (self._dimensions == other._dimensions).all() and \
                self._compound == other._compound:
            raise IncompatibleUnits("add", self, other)

        su = self._udunits()
        ou = other._udunits()
        if not su == ou:
            scaling, offset = _udunits.convert(ou, su)
        units = copy.deepcopy(self._units)
        dimensions = copy.deepcopy(self._dimensions)
        compound = copy.deepcopy(self._compound)
        new = self.__class__(units, dimensions, **compound)
        new._scaling = scaling
        new._offset = offset
        return new

    __sub__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Dimensionality):
            scaling = self.scaling * float(other)
            units = self.units
            dimensions = copy.deepcopy(self._dimensions)
        else:
            units = []
            old = []
            new = []
            for su, ou, od, in zip(self._units, other._units, other._dimensions):
                if (su and ou) and (su != ou):
                    old.append(ou+'^%d'%od)
                    new.append(su+'^%d'%od)
                if ou and not su: units.append(ou)
                else: units.append(su)
            old = '*'.join(old)
            new = '*'.join(new)
            scaling, offset = _udunits.convert(old, new)
            dimensions = self._dimensions + other._dimensions
            compound = copy.deepcopy(self._compound)
            for k, v in other._compound.items():
                if k in self._compound:
                    compound[k] += v
                else:
                    compound[k] = v
        new = self.__class__(units, dimensions, **compound)
        new._scaling = scaling
        new._offset = offset
        return new

    def __div__(self, other):
        if not isinstance(other, Dimensionality):
            scaling = self.scaling / float(other)
            units = self.units
            dimensions = copy.deepcopy(self._dimensions)
        else:
            units = []
            old = []
            new = []
            for su, ou, od, in zip(self._units, other._units, other._dimensions):
                if (su and ou) and (su != ou):
                    old.append(ou+'^%d'%od)
                    new.append(su+'^%d'%od)
                if ou and not su: units.append(ou)
                else: units.append(su)
            old = '*'.join(old)
            new = '*'.join(new)
            scaling, offset = _udunits.convert(old, new)
            dimensions = self._dimensions - other._dimensions
            compound = copy.deepcopy(self._compound)
            for k, v in other._compound.items():
                if k in self._compound:
                    compound[k] += v
                else:
                    compound[k] = v
        new = self.__class__(units, dimensions, **compound)
        new._scaling = scaling
        new._offset = offset
        return new

    def __pow__(self, other):
        compound = {}
        for k, v in self._compound.items():

            compound[k] = v*other
        return self.__class__(copy.copy(self._units),
                              copy.copy(self._dimensions)*float(other),
                              **compound)
