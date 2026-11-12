import numpy as np

from .. import units as pq
from .common import TestCase, unittest

_np_version = tuple(map(int, np.__version__.split(".dev")[0].split(".")))


class TestUmath(TestCase):

    @property
    def q(self):
        return [1,2,3,4] * pq.J

    def test_prod(self):
        self.assertQuantityEqual(np.prod(self.q), 24 * pq.J**4)

    def test_sum(self):
        self.assertQuantityEqual(np.sum(self.q), 10 * pq.J)

    def test_nansum(self):
        c = [1,2,3, np.nan] * pq.m
        self.assertQuantityEqual(np.nansum(c), 6 * pq.m)

    def test_nan_variance_and_standard_deviation(self):
        values = np.array([[1., np.nan, 5.], [2., 4., 8.]])
        for data in (values, values + 1j * values):
            q = data * pq.ms
            for func, units in ((np.nanvar, pq.ms**2), (np.nanstd, pq.ms)):
                for axis in (None, 0, 1):
                    for keepdims in (False, True):
                        with self.subTest(func=func, axis=axis, keepdims=keepdims,
                                          dtype=data.dtype):
                            expected = func(data, axis=axis, keepdims=keepdims)
                            result = func(q, axis=axis, keepdims=keepdims)
                            self.assertQuantityEqual(result, expected * units)
                            out = np.zeros_like(expected) * pq.kg
                            self.assertIs(func(q, axis=axis, keepdims=keepdims,
                                               out=out), out)
                            self.assertQuantityEqual(out, expected * units)
            self.assertQuantityEqual(np.nanstd(q, ddof=1),
                                     np.nanstd(data, ddof=1) * pq.ms)
            np.testing.assert_array_equal(q.magnitude, data)
            self.assertEqual(q.units, pq.ms)

    def test_ufunc_output_units(self):
        q = [2., 4.] * pq.m
        out = [0., 0.] * pq.s
        self.assertIs(np.multiply(q, q, out=out), out)
        self.assertQuantityEqual(out, [4., 16.] * pq.m**2)
        self.assertIs(np.sqrt(out, out=out), out)
        self.assertQuantityEqual(out, q)
        self.assertIs(np.divide(out, q, out=out), out)
        self.assertQuantityEqual(out, [1., 1.] * pq.dimensionless)
        q **= 2
        self.assertQuantityEqual(q, [4., 16.] * pq.m**2)
        scalar = 2. * pq.m
        scalar **= scalar.magnitude
        self.assertQuantityEqual(scalar, 4. * pq.m**2)

    def test_ufunc_output_view_units(self):
        for operation in (
                lambda v: np.multiply(v, v, out=v),
                lambda v: np.divide(v, v, out=v),
                lambda v: np.square(v, out=v),
                lambda v: np.sqrt(v, out=v)):
            with self.subTest(operation=operation):
                q = [1., 4., 9.] * pq.m
                view = q[:2]
                with self.assertRaisesRegex(ValueError, 'modify units of a view'):
                    operation(view)
                self.assertEqual(view.units, pq.m)
                self.assertEqual(q.units, pq.m)

        q = [1., 4., 9.] * pq.m
        view = q[:2]
        self.assertIs(np.multiply(view, 2, out=view), view)
        self.assertQuantityEqual(q, [2., 8., 9.] * pq.m)
        self.assertIs(np.add(view, view, out=view), view)
        self.assertQuantityEqual(q, [4., 16., 9.] * pq.m)
        self.assertQuantityEqual(np.square(view), [16., 256.] * pq.m**2)

        source = [2., 3.] * pq.m
        destination = [0., 0., 0.] * pq.m**2
        output = destination[:2]
        self.assertIs(np.multiply(source, source, out=output), output)
        self.assertQuantityEqual(destination, [4., 9., 0.] * pq.m**2)
        with self.assertRaisesRegex(ValueError, 'modify units of a view'):
            np.sqrt(source, out=output)
        self.assertEqual(destination.units, pq.m**2)
        self.assertEqual(output.units, pq.m**2)

    def test_cumprod(self):
        self.assertRaises(ValueError, np.cumprod, self.q)

        q = [10, .1, 5, 50] * pq.dimensionless
        self.assertQuantityEqual(np.cumprod(q), [10, 1, 5, 250])

    def test_cumsum(self):
        self.assertQuantityEqual(np.cumsum(self.q), [1, 3, 6, 10] * pq.J)

    def test_diff(self):
        self.assertQuantityEqual(np.diff(self.q, 1), [1, 1, 1] * pq.J)

    def test_ediff1d(self):
        self.assertQuantityEqual(np.diff(self.q, 1), [1, 1, 1] * pq.J)

    def test_ediff1d_boundary_units(self):
        from ..quantity import Quantity
        from ..umath import ediff1d

        values = [[1., 3.], [6., 10.]] * pq.m
        for begin, end in ((100 * pq.cm, [200, 300] * pq.cm),
                           (1 * pq.m, [2, 3] * pq.m),
                           (1, [2, 3])):
            with self.subTest(begin=begin, end=end):
                self.assertQuantityEqual(
                    ediff1d(values, to_begin=begin, to_end=end),
                    [1., 2., 3., 4., 2., 3.] * pq.m,
                )
        self.assertQuantityEqual(ediff1d(values, to_begin=100 * pq.cm),
                                 [1., 2., 3., 4.] * pq.m)
        self.assertQuantityEqual(ediff1d(values, to_end=200 * pq.cm),
                                 [2., 3., 4., 2.] * pq.m)
        self.assertRaises(ValueError, ediff1d, values, to_begin=1 * pq.s)
        self.assertRaises(ValueError, ediff1d, values, to_end=1 * pq.s)
        integer_values = Quantity([1, 3, 6], pq.m, dtype=int)
        self.assertQuantityEqual(ediff1d(integer_values, to_begin=1),
                                 [1, 2, 3] * pq.m)
        # Converted fractional boundaries must obey NumPy's casting rules.
        self.assertRaises(TypeError, ediff1d, integer_values,
                          to_begin=1 * pq.cm)
        self.assertQuantityEqual(values, [[1., 3.], [6., 10.]] * pq.m)

    def test_linspace(self):
        self.assertQuantityEqual(np.linspace(self.q[0], self.q[-1], 4), self.q)

    @unittest.expectedFailure
    def test_gradient(self):
        try:
            l = np.gradient([[1,1],[3,4]] * pq.J, 1 * pq.m)
            self.assertQuantityEqual(l[0], [[2., 3.], [2., 3.]] * pq.J/pq.m)
            self.assertQuantityEqual(l[1], [[0., 0.], [1., 1.]] * pq.J/pq.m)
        except ValueError as e:
            raise self.failureException(e)

    @unittest.expectedFailure
    def test_cross(self):
        a = [3,-3, 1] * pq.kPa
        b = [4, 9, 2] * pq.m**2
        self.assertQuantityEqual(np.cross(a,b), [-15,-2,39]*pq.kPa*pq.m**2)

    def test_trapz(self):
        # np.trapz is deprecated, np.trapezoid is recommend
        if _np_version < (2, 0, 0):
            self.assertQuantityEqual(np.trapz(self.q, dx = 1*pq.m), 7.5 * pq.J*pq.m)

    def test_trapezoid(self):
        if _np_version >= (2, 0, 0):
            self.assertQuantityEqual(np.trapezoid(self.q, dx = 1*pq.m), 7.5 * pq.J*pq.m)

    def test_sinh(self):
        q = [1, 2, 3, 4, 6] * pq.radian
        self.assertQuantityEqual(
            np.sinh(q),
            np.sinh(q.magnitude)
            )

    def test_arcsinh(self):
        q = [1, 2, 3, 4, 6] * pq.dimensionless
        self.assertQuantityEqual(
            np.arcsinh(q),
            np.arcsinh(q.magnitude) * pq.rad
            )

    def test_cosh(self):
        q = [1, 2, 3, 4, 6] * pq.radian
        self.assertQuantityEqual(
            np.cosh(q),
            np.cosh(q.magnitude) * pq.dimensionless
            )

    def test_arccosh(self):
        q = [1, 2, 3, 4, 6] * pq.dimensionless
        self.assertQuantityEqual(
            np.arccosh(q),
            np.arccosh(q.magnitude) * pq.rad
            )

    def test_tanh(self):
        q = [1, 2, 3, 4, 6] * pq.rad
        self.assertQuantityEqual(
            np.tanh(q),
            np.tanh(q.magnitude)
            )

    def test_arctanh(self):
        q = [.01, .5, .6, .8, .99] * pq.dimensionless
        self.assertQuantityEqual(
            np.arctanh(q),
            np.arctanh(q.magnitude) * pq.rad
            )

    def test_around(self):
        self.assertQuantityEqual(
            np.around([.5, 1.5, 2.5, 3.5, 4.5] * pq.J) ,
            [0., 2., 2., 4., 4.] * pq.J
            )

        self.assertQuantityEqual(
            np.around([1,2,3,11] * pq.J, decimals=1),
            [1, 2, 3, 11] * pq.J
            )

        self.assertQuantityEqual(
            np.around([1,2,3,11] * pq.J, decimals=-1),
            [0, 0, 0, 10] * pq.J
            )

    def test_round(self):
        self.assertQuantityEqual(
            np.round([.5, 1.5, 2.5, 3.5, 4.5] * pq.J),
            [0., 2., 2., 4., 4.] * pq.J
            )

        self.assertQuantityEqual(
            np.round([1,2,3,11] * pq.J, decimals=1),
            [1, 2, 3, 11] * pq.J
            )

        self.assertQuantityEqual(
            np.round([1,2,3,11] * pq.J, decimals=-1),
            [0, 0, 0, 10] * pq.J
            )

    def test_rint(self):
        a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9] * pq.m
        self.assertQuantityEqual(
            np.rint(a),
            [-4., -4., -2., 0., 2., 3., 4.]*pq.m
            )

    def test_floor(self):
        a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * pq.m
        self.assertQuantityEqual(
            np.floor(a),
            [-2., -2., -1., 0., 1., 1., 2.] * pq.m
            )

    def test_ceil(self):
        a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * pq.m
        self.assertQuantityEqual(
            np.ceil(a),
            [-1., -1., -0., 1., 2., 2., 2.] * pq.m
            )

    def test_fix(self):
        self.assertQuantityEqual(np.fix(3.14 * pq.degF), 3.0 * pq.degF)
        self.assertQuantityEqual(np.fix(3.0 * pq.degF), 3.0 * pq.degF)
        self.assertQuantityEqual(
            np.fix([2.1, 2.9, -2.1, -2.9] * pq.degF),
            [2., 2., -2., -2.] * pq.degF
            )

    def test_exp(self):
        self.assertQuantityEqual(np.exp(1*pq.dimensionless), np.e)
        self.assertRaises(ValueError, np.exp, 1*pq.m)

    def test_exp2(self):
        self.assertQuantityEqual(np.exp2(1*pq.dimensionless), 2.0)
        self.assertRaises(ValueError, np.exp2, 1*pq.m)

    def test_log(self):
        self.assertQuantityEqual(np.log(1*pq.dimensionless), 0)
        self.assertRaises(ValueError, np.log, 1*pq.m)

    def test_log10(self):
        self.assertQuantityEqual(np.log10(1*pq.dimensionless), 0)
        self.assertRaises(ValueError, np.log10, 1*pq.m)

    def test_log2(self):
        self.assertQuantityEqual(np.log2(1*pq.dimensionless), 0)
        self.assertRaises(ValueError, np.log2, 1*pq.m)

    def test_expm1(self):
        self.assertQuantityEqual(np.expm1(1*pq.dimensionless), np.e-1)
        self.assertRaises(ValueError, np.expm1, 1*pq.m)

    def test_log1p(self):
        self.assertQuantityEqual(np.log1p(0*pq.dimensionless), 0)
        self.assertRaises(ValueError, np.log1p, 1*pq.m)

    def test_sin(self):
        self.assertQuantityEqual(np.sin(np.pi/2*pq.radian), 1)
        self.assertRaises(ValueError, np.sin, 1*pq.m)

    def test_arcsin(self):
        self.assertQuantityEqual(
            np.arcsin(1*pq.dimensionless),
            np.pi/2 * pq.radian
            )
        self.assertRaises(ValueError, np.arcsin, 1*pq.m)

    def test_cos(self):
        self.assertQuantityEqual(np.cos(np.pi*pq.radians), -1)
        self.assertRaises(ValueError, np.cos, 1*pq.m)

    def test_arccos(self):
        self.assertQuantityEqual(np.arccos(1*pq.dimensionless), 0*pq.radian)
        self.assertRaises(ValueError, np.arccos, 1*pq.m)

    def test_tan(self):
        self.assertQuantityEqual(np.tan(0*pq.radian), 0)
        self.assertRaises(ValueError, np.tan, 1*pq.m)

    def test_arctan(self):
        self.assertQuantityEqual(np.arctan(0*pq.dimensionless), 0*pq.radian)
        self.assertRaises(ValueError, np.arctan, 1*pq.m)

    def test_arctan2(self):
        self.assertQuantityEqual(
            np.arctan2(0*pq.dimensionless, 0*pq.dimensionless),
            0
            )
        self.assertQuantityEqual(
            np.arctan2(3*pq.V, 3*pq.V),
            np.radians(45)*pq.dimensionless
            )
        # NumPy <1.21 raises ValueError
        # NumPy >=1.21 raises TypeError
        self.assertRaises((TypeError, ValueError), np.arctan2, (1*pq.m, 1*pq.m))

    def test_hypot(self):
        self.assertQuantityEqual(np.hypot(3 * pq.m, 4 * pq.m),  5 * pq.m)
        self.assertRaises(ValueError, np.hypot, 1*pq.m, 2*pq.J)

    def test_degrees(self):
        self.assertQuantityEqual(
            np.degrees(6 * pq.radians),
            (6 * pq.radians).rescale(pq.degree)
            )
        self.assertRaises(ValueError, np.degrees, 0*pq.degree)

    def test_radians(self):
        self.assertQuantityEqual(
            np.radians(6 * pq.degree),
            (6 * pq.degree).rescale(pq.radian)
            )
        self.assertRaises(ValueError, np.radians, 0*pq.radians)

    @unittest.expectedFailure
    def test_unwrap(self):
        self.assertQuantityEqual(np.unwrap([0,3*np.pi]*pq.radians), [0,np.pi])
        self.assertQuantityEqual(np.unwrap([0,540]*pq.deg), [0,180]*pq.deg)

    def test_equal(self):
        arr1 = (1, 1) * pq.m
        arr2 = (1.0, 1.0) * pq.m
        self.assertTrue(np.all(np.equal(arr1, arr2)))
        self.assertFalse(np.all(np.equal(arr1, arr2 * 2)))

    def test_not_equal(self):
        arr1 = (1, 1) * pq.m
        arr2 = (1.0, 1.0) * pq.m
        self.assertTrue(np.all(np.not_equal(arr1, arr2*2)))
        self.assertFalse(np.all(np.not_equal(arr1, arr2)))

    def test_less(self):
        arr1 = (1, 1) * pq.m
        arr2 = (1.0, 1.0) * pq.m
        self.assertTrue(np.all(np.less(arr1, arr2*2)))
        self.assertFalse(np.all(np.less(arr1*2, arr2)))

    def test_less_equal(self):
        arr1 = (1, 1) * pq.m
        arr2 = (1.0, 2.0) * pq.m
        self.assertTrue(np.all(np.less_equal(arr1, arr2)))
        self.assertFalse(np.all(np.less_equal(arr2, arr1)))

    def test_greater(self):
        arr1 = (1, 1) * pq.m
        arr2 = (1.0, 2.0) * pq.m
        self.assertTrue(np.all(np.greater(arr2*1.01, arr1)))
        self.assertFalse(np.all(np.greater(arr2, arr1)))

    def test_greater_equal(self):
        arr1 = (1, 1) * pq.m
        arr2 = (1.0, 2.0) * pq.m
        self.assertTrue(np.all(np.greater_equal(arr2, arr1)))
        self.assertFalse(np.all(np.greater_equal(arr2*0.99, arr1)))

    def test_maximum(self):
        arr1 = (998, 999) * pq.m
        arr2 = (1e3, 5e2) * pq.m
        self.assertQuantityEqual(np.maximum(arr1, arr2) - [1000, 999]*pq.m, [0, 0]*pq.m)

    def test_minimum(self):
        arr1 = (998, 999) * pq.m
        arr2 = (1e3, 5e2) * pq.m
        self.assertQuantityEqual(np.minimum(arr1, arr2) - [998, 500]*pq.m, [0, 0]*pq.m)
