# -*- coding: utf-8 -*-

import pickle

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
from .. import units
from ..uncertainquantity import UncertainQuantity
from .. import constants

from . import assert_quantity_equal, assert_quantity_almost_equal


def test_unitquantitiy_persistance():
    x = units.m
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

    x = units.CompoundUnit("pc/cm**3")
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

def test_quantitiy_persistance():
    x = 20*units.m
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

def test_uncertainquantitiy_persistance():
    x = UncertainQuantity(20, 'm', 0.2)
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

def test_unitconstant_persistance():
    x = constants.m_e
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)
