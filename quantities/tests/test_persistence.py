# -*- coding: utf-8 -*-

import pickle

from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase


class TestPersistence(TestCase):

    def test_unitquantity_persistence(self):
        x = pq.m
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

        x = pq.CompoundUnit("pc/cm**3")
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_quantity_persistence(self):
        x = 20*pq.m
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_uncertainquantity_persistence(self):
        x = UncertainQuantity(20, 'm', 0.2)
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_unitconstant_persistence(self):
        x = constants.m_e
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_quantity_object_dtype(self):
        # Regression test for github issue #113
        x = Quantity(1,dtype=object)
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_uncertainquantity_object_dtype(self):
        # Regression test for github issue #113
        x = UncertainQuantity(20, 'm', 0.2, dtype=object)
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

