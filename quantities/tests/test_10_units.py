# -*- coding: utf-8 -*-

from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
from .. import units


def test_units_protected():
    def setunits(u, v):
        u.units = v
    def inplace(op, u, val):
        getattr(u, '__i%s__'%op)(val)
    assert_raises(AttributeError, setunits, units.m, units.ft)
    assert_raises(TypeError, inplace, 'add', units.m, units.m)
    assert_raises(TypeError, inplace, 'sub', units.m, units.m)
    assert_raises(TypeError, inplace, 'mul', units.m, units.m)
    assert_raises(TypeError, inplace, 'truediv', units.m, units.m)
    assert_raises(TypeError, inplace, 'pow', units.m, 2)
