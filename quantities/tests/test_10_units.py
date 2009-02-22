# -*- coding: utf-8 -*-

from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq


def test_units_protected():
    def setunits(u, v):
        u.units = v
    def inplace(op, u, val):
        getattr(u, '__i%s__'%op)(val)
    assert_raises(AttributeError, setunits, pq.m, pq.ft)
    assert_raises(TypeError, inplace, 'add', pq.m, pq.m)
    assert_raises(TypeError, inplace, 'sub', pq.m, pq.m)
    assert_raises(TypeError, inplace, 'mul', pq.m, pq.m)
    assert_raises(TypeError, inplace, 'div', pq.m, pq.m)
    assert_raises(TypeError, inplace, 'truediv', pq.m, pq.m)
    assert_raises(TypeError, inplace, 'pow', pq.m, 2)
