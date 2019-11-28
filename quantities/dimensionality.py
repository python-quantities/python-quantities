# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

import sys
import operator

import numpy as np

from . import markup
from .registry import unit_registry
from .decorators import memoize

def assert_isinstance(obj, types):
    try:
        assert isinstance(obj, types)
    except AssertionError:
        raise TypeError(
            "arg %r must be of type %r, got %r" % (obj, types, type(obj))
        )


class Dimensionality(dict):

    """
    """

    @property
    def ndims(self):
        return sum(abs(i) for i in self.simplified.values())

    @property
    def simplified(self):
        if len(self):
            rq = 1*unit_registry['dimensionless']
            for u, d in self.items():
                rq = rq * u.simplified**d
            return rq.dimensionality
        else:
            return self

    @property
    def string(self):
        return markup.format_units(self)

    @property
    def unicode(self):
        return markup.format_units_unicode(self)

    @property
    def latex(self):
        return markup.format_units_latex(self)

    @property
    def html(self):
        return markup.format_units_html(self)

    def __hash__(self):
        res = hash(unit_registry['dimensionless'])
        for key in sorted(self.keys(), key=operator.attrgetter('format_order')):
            val = self[key]
            if val < 0:
                # can you believe that hash(-1)==hash(-2)?
                val -= 1
            res ^= hash((key, val))
        return res

    def __add__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self.copy()

    __radd__ = __add__

    def __iadd__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self

    def __sub__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not subtract units of %s and %s'\
                %(str(self), str(other))
            )
        return self.copy()

    __rsub__ = __sub__

    def __isub__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError(
                'can not add units of %s and %s'\
                %(str(self), str(other))
            )
        return self

    def __mul__(self, other):
        assert_isinstance(other, Dimensionality)
        new = Dimensionality(self)
        for unit, power in other.items():
            try:
                new[unit] += power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = power
        return new

    def __imul__(self, other):
        assert_isinstance(other, Dimensionality)
        for unit, power in other.items():
            try:
                self[unit] += power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = power
        return self

    def __truediv__(self, other):
        assert_isinstance(other, Dimensionality)
        new = Dimensionality(self)
        for unit, power in other.items():
            try:
                new[unit] -= power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = -power
        return new

    if sys.version_info[0] < 3:
        def __div__(self, other):
            assert_isinstance(other, Dimensionality)
            return self.__truediv__(other)

    def __itruediv__(self, other):
        assert_isinstance(other, Dimensionality)
        for unit, power in other.items():
            try:
                self[unit] -= power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = -power
        return self

    if sys.version_info[0] < 3:
        def __idiv__(self, other):
            assert_isinstance(other, Dimensionality)
            return self.__itruediv__(other)

    def __pow__(self, other):
        try:
            assert np.isscalar(other)
        except AssertionError:
            raise TypeError('exponent must be a scalar, got %r' % other)
        if other == 0:
            return Dimensionality()
        new = Dimensionality(self)
        for i in new:
            new[i] *= other
        return new

    def __ipow__(self, other):
        try:
            assert np.isscalar(other)
        except AssertionError:
            raise TypeError('exponent must be a scalar, got %r' % other)
        if other == 0:
            self.clear()
            return self
        for i in self:
            self[i] *= other
        return self

    def __repr__(self):
        return 'Dimensionality({%s})' \
            % ', '.join(['%s: %s'% (u.name, e) for u, e in self.items()])

    def __str__(self):
        if markup.config.use_unicode:
            return self.unicode
        else:
            return self.string

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return hash(self) != hash(other)

    __neq__ = __ne__

    def __gt__(self, other):
        return self.ndims > other.ndims

    def __ge__(self, other):
        return self.ndims >= other.ndims

    def __lt__(self, other):
        return self.ndims < other.ndims

    def __le__(self, other):
        return self.ndims <= other.ndims

    def copy(self):
        return Dimensionality(self)


p_dict = {}

def _d_multiply(q1, q2, out=None):
    try:
        return q1._dimensionality * q2._dimensionality
    except AttributeError:
        try:
            return q1.dimensionality
        except:
            return q2.dimensionality
p_dict[np.multiply] = _d_multiply
p_dict[np.cross] = _d_multiply

def _d_divide(q1, q2, out=None):
    try:
        return q1._dimensionality / q2._dimensionality
    except AttributeError:
        try:
            return q1.dimensionality
        except:
            return q2.dimensionality**-1
p_dict[np.divide] = _d_divide
p_dict[np.true_divide] = _d_divide

def _d_check_uniform(q1, q2, out=None):
    try:
        assert q1._dimensionality == q2._dimensionality
        return q1.dimensionality
    except AssertionError:
        raise ValueError(
            'quantities must have identical units, got "%s" and "%s"' %
            (q1.units, q2.units)
        )
    except AttributeError:
        try:
            if hasattr(q1, 'dimensionality'):
                # q2 was not a quantity
                if not q1._dimensionality or not np.asarray(q2).any():
                    return q1.dimensionality
                else:
                    raise ValueError
            elif hasattr(q2, 'dimensionality'):
                # q1 was not a quantity
                if not q2._dimensionality or not np.asarray(q1).any():
                    return q2.dimensionality
                else:
                    raise ValueError
        except ValueError:
            raise ValueError(
                'quantities must have identical units, got "%s" and "%s"' %
                (q1.units, q2.units)
            )

p_dict[np.add] = _d_check_uniform
p_dict[np.subtract] = _d_check_uniform
p_dict[np.mod] = _d_check_uniform
p_dict[np.fmod] = _d_check_uniform
p_dict[np.remainder] = _d_check_uniform
p_dict[np.floor_divide] = _d_check_uniform
p_dict[np.hypot] = _d_check_uniform
p_dict[np.equal] = _d_check_uniform
p_dict[np.not_equal] = _d_check_uniform
p_dict[np.less] = _d_check_uniform
p_dict[np.less_equal] = _d_check_uniform
p_dict[np.greater] = _d_check_uniform
p_dict[np.greater_equal] = _d_check_uniform

def _d_arctan2(q1, q2, out=None):
    try:
        assert q1._dimensionality == q2._dimensionality
        return Dimensionality()
    except AssertionError:
        raise ValueError(
            'quantities must have identical units, got "%s" and "%s"' %
            (q1.units, q2.units)
        )

p_dict[np.arctan2] = _d_arctan2

def _d_power(q1, q2, out=None):
    if getattr(q2, 'dimensionality', None):
        raise ValueError("exponent must be dimensionless")
    try:
        q2 = np.array(q2)
        p = q2.min()
        if p != q2.max():
            raise ValueError('Quantities must be raised to a uniform power')
        return q1._dimensionality**p
    except AttributeError:
        return Dimensionality()
p_dict[np.power] = _d_power

def _d_square(q1, out=None):
    return q1._dimensionality**2
p_dict[np.square] = _d_square

def _d_reciprocal(q1, out=None):
    return q1._dimensionality**-1
p_dict[np.reciprocal] = _d_reciprocal

def _d_copy(q1, out=None):
    return q1.dimensionality
p_dict[np.fabs] = _d_copy
p_dict[np.absolute] = _d_copy
p_dict[np.conjugate] = _d_copy
p_dict[np.negative] = _d_copy
p_dict[np.ones_like] = _d_copy
p_dict[np.rint] = _d_copy
p_dict[np.floor] = _d_copy
p_dict[np.fix] = _d_copy
p_dict[np.ceil] = _d_copy

def _d_clip(a1, a2, a3, q):
    return q.dimensionality
try:
    p_dict[np.core.umath.clip] = _d_clip
except AttributeError:
    pass # For compatibility with Numpy < 1.17 when clip wasn't a ufunc yet

def _d_sqrt(q1, out=None):
    return q1._dimensionality**0.5
p_dict[np.sqrt] = _d_sqrt

def _d_radians(q1, out=None):
    try:
        assert q1.units == unit_registry['degree']
    except AssertionError:
        raise ValueError(
            'expected units of degrees, got "%s"' % q1._dimensionality
        )
    return unit_registry['radian'].dimensionality
p_dict[np.radians] = _d_radians

def _d_degrees(q1, out=None):
    try:
        assert q1.units == unit_registry['radian']
    except AssertionError:
        raise ValueError(
            'expected units of radians, got "%s"' % q1._dimensionality
        )
    return unit_registry['degree'].dimensionality
p_dict[np.degrees] = _d_degrees

def _d_dimensionless(q1, out=None):
    if getattr(q1, 'dimensionality', None):
        raise ValueError("quantity must be dimensionless")
    return Dimensionality()
p_dict[np.log] = _d_dimensionless
p_dict[np.log10] = _d_dimensionless
p_dict[np.log2] = _d_dimensionless
p_dict[np.log1p] = _d_dimensionless
p_dict[np.exp] = _d_dimensionless
p_dict[np.expm1] = _d_dimensionless
p_dict[np.logaddexp] = _d_dimensionless
p_dict[np.logaddexp2] = _d_dimensionless

def _d_trig(q1, out=None):
    try:
        assert q1.units == unit_registry['radian']
    except AssertionError:
        raise ValueError(
            'expected units of radians, got "%s"' % q1._dimensionality
        )
    return Dimensionality()
p_dict[np.sin] = _d_trig
p_dict[np.sinh] = _d_trig
p_dict[np.cos] = _d_trig
p_dict[np.cosh] = _d_trig
p_dict[np.tan] = _d_trig
p_dict[np.tanh] = _d_trig

def _d_arctrig(q1, out=None):
    if getattr(q1, 'dimensionality', None):
        raise ValueError("quantity must be dimensionless")
    return unit_registry['radian'].dimensionality
p_dict[np.arcsin] = _d_arctrig
p_dict[np.arcsinh] = _d_arctrig
p_dict[np.arccos] = _d_arctrig
p_dict[np.arccosh] = _d_arctrig
p_dict[np.arctan] = _d_arctrig
p_dict[np.arctanh] = _d_arctrig
