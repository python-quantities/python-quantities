# -*- coding: utf-8 -*-

import unittest

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as q


def test_units_protected():
    def setunits(u, v):
        u.units = v
    def inplace(op, u, val):
        getattr(u, '__i%s__'%op)(val)
    assert_raises(AttributeError, setunits, q.m, q.ft)
    assert_raises(TypeError, inplace, 'add', q.m, q.m)
    assert_raises(TypeError, inplace, 'sub', q.m, q.m)
    assert_raises(TypeError, inplace, 'mul', q.m, q.m)
    assert_raises(TypeError, inplace, 'div', q.m, q.m)
    assert_raises(TypeError, inplace, 'truediv', q.m, q.m)
    assert_raises(TypeError, inplace, 'pow', q.m, 2)
