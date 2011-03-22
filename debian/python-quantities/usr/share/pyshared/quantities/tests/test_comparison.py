# -*- coding: utf-8 -*-

import operator as op

from .. import units as pq
from .common import TestCase


class TestComparison(TestCase):

    def test_scalar_equality(self):
        self.assertEqual(pq.J == pq.J, [True])
        self.assertEqual(1*pq.J == pq.J, [True])
        self.assertEqual(str(1*pq.J) == '1.0 J', True)
        self.assertEqual(pq.J == pq.kg*pq.m**2/pq.s**2, [True])

        self.assertEqual(pq.J == pq.erg, [False])
        self.assertEqual(2*pq.J == pq.J, [False])
        self.assertEqual(pq.J == 2*pq.kg*pq.m**2/pq.s**2, [False])

        self.assertEqual(pq.J == pq.kg, [False])

    def test_scalar_inequality(self):
        self.assertEqual(pq.J != pq.erg, [True])
        self.assertEqual(2*pq.J != pq.J, [True])
        self.assertEqual(str(2*pq.J) != str(pq.J), True)
        self.assertEqual(pq.J != 2*pq.kg*pq.m**2/pq.s**2, [True])

        self.assertEqual(pq.J != pq.J, [False])
        self.assertEqual(1*pq.J != pq.J, [False])
        self.assertEqual(pq.J != 1*pq.kg*pq.m**2/pq.s**2, [False])

    def test_scalar_comparison(self):
        self.assertEqual(2*pq.J > pq.J, [True])
        self.assertEqual(2*pq.J > 1*pq.J, [True])
        self.assertEqual(1*pq.J >= pq.J, [True])
        self.assertEqual(1*pq.J >= 1*pq.J, [True])
        self.assertEqual(2*pq.J >= pq.J, [True])
        self.assertEqual(2*pq.J >= 1*pq.J, [True])

        self.assertEqual(0.5*pq.J < pq.J, [True])
        self.assertEqual(0.5*pq.J < 1*pq.J, [True])
        self.assertEqual(0.5*pq.J <= pq.J, [True])
        self.assertEqual(0.5*pq.J <= 1*pq.J, [True])
        self.assertEqual(1.0*pq.J <= pq.J, [True])
        self.assertEqual(1.0*pq.J <= 1*pq.J, [True])

        self.assertEqual(2*pq.J < pq.J, [False])
        self.assertEqual(2*pq.J < 1*pq.J, [False])
        self.assertEqual(2*pq.J <= pq.J, [False])
        self.assertEqual(2*pq.J <= 1*pq.J, [False])

        self.assertEqual(0.5*pq.J > pq.J, [False])
        self.assertEqual(0.5*pq.J > 1*pq.J, [False])
        self.assertEqual(0.5*pq.J >= pq.J, [False])
        self.assertEqual(0.5*pq.J >= 1*pq.J, [False])

    def test_array_equality(self):
        self.assertQuantityEqual(
            [1, 2, 3, 4]*pq.J == [1, 22, 3, 44]*pq.J,
            [1, 0, 1, 0]
        )
        self.assertQuantityEqual(
            [1, 2, 3, 4]*pq.J == [1, 22, 3, 44]*pq.kg,
            [0, 0, 0, 0]
        )
        self.assertQuantityEqual(
            [1, 2, 3, 4]*pq.J == [1, 22, 3, 44],
            [1, 0, 1, 0]
        )

    def test_array_inequality(self):
        self.assertQuantityEqual(
            [1, 2, 3, 4]*pq.J != [1, 22, 3, 44]*pq.J,
            [0, 1, 0, 1]
        )
        self.assertQuantityEqual(
            [1, 2, 3, 4]*pq.J != [1, 22, 3, 44]*pq.kg,
            [1, 1, 1, 1]
        )
        self.assertQuantityEqual(
            [1, 2, 3, 4]*pq.J != [1, 22, 3, 44],
            [0, 1, 0, 1]
        )

    def test_quantity_less_than(self):
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J < [1, 22, 3]*pq.J,
            [0, 1, 0]
        )
        self.assertQuantityEqual(
            [50, 100, 150]*pq.cm < [1, 1, 1]*pq.m,
            [1, 0, 0]
        )
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J < [1, 22, 3],
            [0, 1, 0]
        )
        self.assertRaises(
            ValueError,
            op.lt,
            [1, 2, 33]*pq.J,
            [1, 22, 3]*pq.kg,
        )

    def test_quantity_less_than_or_equal(self):
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J <= [1, 22, 3]*pq.J,
            [1, 1, 0]
        )
        self.assertQuantityEqual(
            [50, 100, 150]*pq.cm <= [1, 1, 1]*pq.m,
            [1, 1, 0]
        )
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J <= [1, 22, 3],
            [1, 1, 0]
        )
        self.assertRaises(
            ValueError,
            op.le,
            [1, 2, 33]*pq.J,
            [1, 22, 3]*pq.kg,
        )

    def test_quantity_greater_than_or_equal(self):
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J >= [1, 22, 3]*pq.J,
            [1, 0, 1]
        )
        self.assertQuantityEqual(
            [50, 100, 150]*pq.cm >= [1, 1, 1]*pq.m,
            [0, 1, 1]
        )
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J >= [1, 22, 3],
            [1, 0, 1]
        )
        self.assertRaises(
            ValueError,
            op.ge,
            [1, 2, 33]*pq.J,
            [1, 22, 3]*pq.kg,
        )

    def test_quantity_greater_than(self):
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J > [1, 22, 3]*pq.J,
            [0, 0, 1]
        )
        self.assertQuantityEqual(
            [50, 100, 150]*pq.cm > [1, 1, 1]*pq.m,
            [0, 0, 1]
        )
        self.assertQuantityEqual(
            [1, 2, 33]*pq.J > [1, 22, 3],
            [0, 0, 1]
        )
        self.assertRaises(
            ValueError,
            op.gt,
            [1, 2, 33]*pq.J,
            [1, 22, 3]*pq.kg,
        )
