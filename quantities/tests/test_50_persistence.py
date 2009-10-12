# -*- coding: utf-8 -*-

import pickle

from nose.tools import *
from numpy.testing import *
from numpy.testing.utils import *

import numpy as np
import quantities as pq

from . import assert_quantity_equal, assert_quantity_almost_equal


def test_quantitiy_persistance():
    a = 20*pq.m
    temp = pickle.dumps(a)
    b = pickle.loads(temp)
    assert_array_equal(a, b)



if __name__ == "__main__":
    run_module_suite()
