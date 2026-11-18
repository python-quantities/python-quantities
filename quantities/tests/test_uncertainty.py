import numpy as np

from .. import units as pq
from ..quantity import Quantity
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
        expected = [3.2808399, 3.2808399, 3.2808399]*pq.ft
        self.assertQuantityEqual(a.rescale(pq.ft), expected)
        self.assertQuantityEqual(a.rescale('ft'), expected)
        self.assertQuantityEqual(
            a.rescale('ft').uncertainty,
            [0.32808399, 0.32808399, 0.32808399]*pq.ft
            )

        seventy_km = Quantity(70, pq.km, dtype=np.float32)
        seven_km = Quantity(7, pq.km, dtype=np.float32)
        seventyish_km = UncertainQuantity(seventy_km, pq.km, seven_km, dtype=np.float32)
        self.assertTrue(seventyish_km.dtype == np.float32)
        in_meters = seventyish_km.rescale(pq.m)
        self.assertTrue(in_meters.dtype == seventyish_km.dtype)
        seventyish_km_rescaled_idempotent = seventyish_km.rescale(pq.km)
        self.assertTrue(seventyish_km_rescaled_idempotent.dtype == np.float32)
        self.assertQuantityEqual(seventyish_km + in_meters, 2*seventy_km)

    def test_creation_from_uncertain_quantity(self):
        for magnitude, uncertainty in [(1, 0.1), ([1, 2], [0.1, 0.2])]:
            source = UncertainQuantity(magnitude, pq.m, uncertainty)
            for units in ['', pq.m, pq.cm]:
                with self.subTest(magnitude=magnitude, units=str(units)):
                    result = UncertainQuantity(source, units)
                    expected_units = units or pq.m
                    self.assertQuantityEqual(result, Quantity(magnitude, pq.m).rescale(expected_units))
                    self.assertQuantityEqual(
                        result.uncertainty, Quantity(uncertainty, pq.m).rescale(expected_units)
                    )
                    self.assertEqual(result.dimensionality, Quantity(1, expected_units).dimensionality)
                    self.assertEqual(result.uncertainty.dimensionality, result.dimensionality)
                    self.assertQuantityEqual(source, Quantity(magnitude, pq.m))
                    self.assertQuantityEqual(source.uncertainty, Quantity(uncertainty, pq.m))

    def test_creation_from_uncertain_quantity_with_explicit_uncertainty(self):
        source = UncertainQuantity([1, 2], pq.m, [0.1, 0.2])
        result = UncertainQuantity(source, pq.cm, uncertainty=[3, 4])
        self.assertQuantityEqual(result, [100, 200] * pq.cm)
        self.assertQuantityEqual(result.uncertainty, [3, 4] * pq.cm)
        self.assertQuantityEqual(source.uncertainty, [0.1, 0.2] * pq.m)


    def test_set_uncertainty(self):
        a = UncertainQuantity([1, 2], 'm', [.1, .2])
        a.uncertainty = [1., 2.]*pq.m
        self.assertQuantityEqual(a.uncertainty, [1, 2]*pq.m)

        def set_u(q, u):
            q.uncertainty = u

        self.assertRaises(ValueError, set_u, a, 1)
        self.assertRaises(ValueError, set_u, a, [1,2])

    def test_uncertainquantity_add_subtract_exact(self):
        # an operand without an uncertainty is treated as exact (#45)
        a = UncertainQuantity([1., 2.], pq.m, [.1, .2])
        cases = [
            (lambda: a + 1*pq.m, [2., 3.]),
            (lambda: 1*pq.m + a, [2., 3.]),
            (lambda: a - 1*pq.cm, [.99, 1.99]),
            (lambda: 1*pq.cm - a, [-.99, -1.99]),
            (lambda: a - [1., 1.]*pq.m, [0., 1.]),
        ]
        for i, (f, expected) in enumerate(cases):
            with self.subTest(case=i):
                res = f()
                self.assertIsInstance(res, UncertainQuantity)
                self.assertQuantityEqual(res, expected*pq.m)
                self.assertQuantityEqual(res.uncertainty, [.1, .2]*pq.m)

    def test_uncertainquantity_add_subtract_dimensionless(self):
        u = UncertainQuantity(1, pq.percent, .1)
        cases = [(lambda: u + 1, 101), (lambda: 1 + u, 101), (lambda: u - 1, -99), (lambda: 1 - u, 99)]
        for i, (f, expected) in enumerate(cases):
            with self.subTest(case=i):
                res = f()
                self.assertIsInstance(res, UncertainQuantity)
                self.assertQuantityEqual(res, expected*pq.percent)
                self.assertQuantityEqual(res.uncertainty, .1*pq.percent)

    def test_uncertainquantity_add_broadcast_exact(self):
        d = UncertainQuantity(2., pq.dimensionless, .1)
        res = d + np.array([1., 2.])
        self.assertQuantityEqual(res, [3., 4.]*pq.dimensionless)
        self.assertQuantityEqual(res.uncertainty, [.1, .1]*pq.dimensionless)

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
