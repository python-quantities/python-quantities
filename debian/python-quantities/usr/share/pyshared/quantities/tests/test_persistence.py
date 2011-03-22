# -*- coding: utf-8 -*-

import pickle

from .. import units as pq
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase


class TestPersistence(TestCase):

    def test_unitquantity_persistance(self):
        x = pq.m
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

        x = pq.CompoundUnit("pc/cm**3")
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_quantity_persistance(self):
        x = 20*pq.m
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_uncertainquantity_persistance(self):
        x = UncertainQuantity(20, 'm', 0.2)
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_unitconstant_persistance(self):
        x = constants.m_e
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)
