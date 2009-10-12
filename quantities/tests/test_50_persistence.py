# -*- coding: utf-8 -*-

import pickle

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq

from . import assert_quantity_equal, assert_quantity_almost_equal


def test_unitquantitiy_persistance():
    x = pq.m
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

    x = pq.CompoundUnit("pc/cm**3")
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

def test_quantitiy_persistance():
    x = 20*pq.m
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

def test_uncertainquantitiy_persistance():
    x = pq.UncertainQuantity(20, 'm', 0.2)
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)

def test_unitconstant_persistance():
    x = pq.constants.m_e
    y = pickle.loads(pickle.dumps(x))
    assert_array_equal(x, y)


if __name__ == "__main__":
    run_module_suite()
