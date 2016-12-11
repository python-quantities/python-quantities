# -*- coding: utf-8 -*-

from .. import units as pq
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np


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

    def test_uncertainquantity_negative(self):
        a = UncertainQuantity([1, 2], 'm', [.1, .2])
        self.assertQuantityEqual(-a, [-1., -2.]*pq.m)
        self.assertQuantityEqual((-a).uncertainty, [0.1, 0.2]*pq.m)
        self.assertQuantityEqual(-a, a*-1)
        self.assertQuantityEqual((-a).uncertainty, (a*-1).uncertainty)

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

    def test_uncertaintity_mean(self):
        a = UncertainQuantity([1,2], 'm', [.1,.2])
        mean0 = np.sum(a)/np.size(a) # calculated traditionally
        mean1 = a.mean()        # calculated using this code
        self.assertQuantityEqual(mean0, mean1)

    def test_uncertaintity_nanmean(self):
        a = UncertainQuantity([1,2], 'm', [.1,.2])
        b = UncertainQuantity([1,2,np.nan], 'm', [.1,.2,np.nan])
        self.assertQuantityEqual(a.mean(),b.nanmean())

    def test_uncertainty_sqrt(self):
        a = UncertainQuantity([1,2], 'm', [.1,.2])
        self.assertQuantityEqual(a**0.5, a.sqrt())

    def test_uncertainty_nansum(self):
        uq = UncertainQuantity([1,2], 'm', [1,1])
        uq_nan = UncertainQuantity([1,2,np.nan], 'm', [1,1,np.nan])
        self.assertQuantityEqual(np.sum(uq), np.nansum(uq))
        self.assertQuantityEqual(np.sum(uq), uq_nan.nansum())

    def test_uncertainty_minmax_nan_arg(self):
        q = [[1, 2], [3, 4]] * pq.m        # quantity
        self.assertQuantityEqual(q.min(), 1*pq.m) # min
        self.assertQuantityEqual(q.max(), 4*pq.m) # max
        self.assertQuantityEqual(q.argmin(), 0) # argmin
        self.assertQuantityEqual(q.argmax(), 3) # argmax
        # uncertain quantity
        uq = UncertainQuantity([[1,2],[3,4]], pq.m, [[1,1],[1,1]])
        self.assertQuantityEqual(uq.min(), 1*pq.m) # min
        self.assertQuantityEqual(uq.max(), 4*pq.m) # max
        self.assertQuantityEqual(uq.argmin(), 0) # argmin
        self.assertQuantityEqual(uq.argmax(), 3) # argmax
        # now repeat the above with NaNs
        nanq = [[1, 2], [3, 4], [np.nan,np.nan]] * pq.m        # quantity
        nanuq = UncertainQuantity([[1,2],[3,4],[np.nan,np.nan]],
                               pq.m,
                               [[1,1],[1,1],[np.nan,np.nan]])
        self.assertQuantityEqual(nanq.nanmin(), 1*pq.m) # min
        self.assertQuantityEqual(nanq.nanmax(), 4*pq.m) # max
        self.assertQuantityEqual(nanq.nanargmin(), 0) # argmin
        self.assertQuantityEqual(nanq.nanargmax(), 3) # argmax
        self.assertQuantityEqual(nanuq.nanmin(), 1*pq.m) # min
        self.assertQuantityEqual(nanuq.nanmax(), 4*pq.m) # max
        self.assertQuantityEqual(nanuq.nanargmin(), 0) # argmin
        self.assertQuantityEqual(nanuq.nanargmax(), 3) # argmax
