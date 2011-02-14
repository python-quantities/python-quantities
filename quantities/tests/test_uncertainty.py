# -*- coding: utf-8 -*-

from .. import units as pq
from ..uncertainquantity import UncertainQuantity
from .common import TestCase


class TestUncertainty(TestCase):

    def test_creation(self):
        a = UncertainQuantity(1, pq.m)
        self.assertQuantityEqual(a, 1*pq.m)
        self.assertQuantityEqual(a.uncertainty, 0*pq.m)
        a = UncertainQuantity([1, 1, 1], pq.m)
        self.assertQuantityEqual(a, [1,1,1]*pq.m)
        self.assertQuantityEqual(a.uncertainty, [0,0,0]*pq.m)
        a = UncertainQuantity([1, 1, 1], pq.m, [.1, .1, .1])
        self.assertQuantityEqual(a, [1, 1, 1] *pq.m)
        self.assertQuantityEqual(a.uncertainty, [0.1, 0.1, 0.1] *pq.m)
        self.assertRaises(ValueError, UncertainQuantity, [1,1,1], pq.m, 1)
        self.assertRaises(ValueError, UncertainQuantity, [1,1,1], pq.m, [1,1])

    def test_rescale(self):
        a = UncertainQuantity([1, 1, 1], pq.m, [.1, .1, .1])
        b = a.rescale(pq.ft)
        self.assertQuantityEqual(
            a.rescale('ft'),
            [3.2808399, 3.2808399, 3.2808399]*pq.ft
            )
        self.assertQuantityEqual(
            a.rescale('ft').uncertainty,
            [0.32808399, 0.32808399, 0.32808399]*pq.ft
            )

    def test_set_uncertainty(self):
        a = UncertainQuantity([1, 2], 'm', [.1, .2])
        a.uncertainty = [1., 2.]*pq.m
        self.assertQuantityEqual(a.uncertainty, [1, 2]*pq.m)

        def set_u(q, u):
            q.uncertainty = u

        self.assertRaises(ValueError, set_u, a, 1)
        self.assertRaises(ValueError, set_u, a, [1,2])

    def test_uncertainquantity_multiply(self):
        a = UncertainQuantity([1, 2], 'm', [.1, .2])
        self.assertQuantityEqual(a*a, [1., 4.]*pq.m**2)
        self.assertQuantityEqual((a*a).uncertainty, [0.14142,0.56568]*pq.m**2)
        self.assertQuantityEqual(a*2, [2, 4]*pq.m)
        self.assertQuantityEqual((a*2).uncertainty, [0.2,0.4]*pq.m)

    def test_uncertainquantity_divide(self):
        a = UncertainQuantity([1, 2], 'm', [.1, .2])
        self.assertQuantityEqual(a/a, [1., 1.])
        self.assertQuantityEqual((a/a).uncertainty, [0.14142, 0.14142])
        self.assertQuantityEqual(a/pq.m, [1., 2.])
        self.assertQuantityEqual((a/pq.m).uncertainty, [0.1, 0.2])
        self.assertQuantityEqual(a/2, [0.5, 1.]*pq.m)
        self.assertQuantityEqual((a/2).uncertainty, [0.05, 0.1 ]*pq.m)
        self.assertQuantityEqual(1/a, [1., 0.5]/pq.m)
        self.assertQuantityEqual((1/a).uncertainty, [0.1, 0.05]/pq.m)
