# -*- coding: utf-8 -*-

from ..quantity import Quantity
from .test_methods import TestQuantityMethods, TestCase

class ChildQuantity(Quantity):
  def __new__(cls, data, units='', dtype=None, copy=True):
    obj = Quantity.__new__(cls, data, units, dtype, copy).view(cls)
    return obj

class TestQuantityInheritance(TestCase):

  def setUp(self):
    self.cq = ChildQuantity([1,5], '')

  def test_resulting_type(self):
    self.assertTrue (isinstance(self.cq, ChildQuantity))
    self.assertTrue (isinstance(self.cq + self.cq, ChildQuantity))
    self.assertTrue (isinstance(self.cq * self.cq, ChildQuantity))
    self.assertTrue (isinstance(self.cq / self.cq, ChildQuantity))
    self.assertTrue (isinstance(self.cq - self.cq, ChildQuantity))
    self.assertTrue (isinstance(self.cq.max(), ChildQuantity))
    self.assertTrue (isinstance(self.cq.min(), ChildQuantity))
    self.assertTrue (isinstance(self.cq.mean(), ChildQuantity))
    self.assertTrue (isinstance(self.cq.var(), ChildQuantity))
    self.assertTrue (isinstance(self.cq.std(), ChildQuantity))
    self.assertTrue (isinstance(self.cq.prod(), ChildQuantity))
    self.assertTrue (isinstance(self.cq.cumsum(), ChildQuantity))
    self.assertTrue (isinstance(self.cq.cumprod(), ChildQuantity))
