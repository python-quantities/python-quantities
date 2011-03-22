# -*- coding: utf-8 -*-

from .. import units as pq
from .common import TestCase


class TestQuantityMethods(TestCase):

    def setUp(self):
        self.q = [[1, 2], [3, 4]] * pq.m

    def test_tolist(self):
        self.assertEqual(self.q.tolist(), [[1*pq.m, 2*pq.m], [3*pq.m, 4*pq.m]])

    def test_sum(self):
        self.assertQuantityEqual(self.q.sum(), 10*pq.m)
        self.assertQuantityEqual(self.q.sum(0), [4, 6]*pq.m)
        self.assertQuantityEqual(self.q.sum(1), [3, 7]*pq.m)

    def test_fill(self):
        self.q.fill(6 * pq.ft)
        self.assertQuantityEqual(self.q, [[6, 6], [6, 6]] * pq.ft)
        self.q.fill(5)
        self.assertQuantityEqual(self.q, [[5, 5], [5, 5]] * pq.ft)

    def test_reshape(self):
        self.assertQuantityEqual(self.q.reshape([1,4]), [[1, 2, 3, 4]] * pq.m)

    def test_transpose(self):
        self.assertQuantityEqual(self.q.transpose(), [[1, 3], [2, 4]] * pq.m)

    def test_flatten(self):
        self.assertQuantityEqual(self.q.flatten(), [1, 2, 3, 4] * pq.m)

    def test_ravel(self):
        self.assertQuantityEqual(self.q.ravel(), [1, 2, 3, 4] * pq.m)

    def test_squeeze(self):
        self.assertQuantityEqual(
            self.q.reshape([1,4]).squeeze(),
            [1, 2, 3, 4] * pq.m
            )

    def test_take(self):
        self.assertQuantityEqual(self.q.take([0,1,2,3]), self.q.flatten())

    def test_put(self):
        q = self.q.flatten()
        q.put([0,2], [10,20]*pq.m)
        self.assertQuantityEqual(q, [10, 2, 20, 4]*pq.m)

        q = self.q.flatten()
        q.put([0, 2], [1, 2]*pq.mm)
        self.assertQuantityEqual(q, [0.001, 2, 0.002, 4]*pq.m)

        q = self.q.flatten()/pq.mm
        q.put([0, 2], [1, 2])
        self.assertQuantityEqual(q.simplified, [1, 2000, 2, 4000])
        self.assertQuantityEqual(q, [0.001, 2, 0.002, 4]*pq.m/pq.mm)

        q = self.q.flatten()
        self.assertRaises(ValueError, q.put, [0, 2], [4, 6] * pq.J)
        self.assertRaises(ValueError, q.put, [0, 2], [4, 6])

    def test_repeat(self):
        self.assertQuantityEqual(self.q.repeat(2), [1,1,2,2,3,3,4,4]*pq.m)

    def test_sort(self):
        q = [4, 5, 2, 3, 1, 6] * pq.m
        q.sort()
        self.assertQuantityEqual(q, [1, 2, 3, 4, 5, 6] * pq.m)

    def test_argsort(self):
        q = [1, 4, 5, 6, 2, 9] * pq.MeV
        self.assertQuantityEqual(q.argsort(), [0, 4, 1, 2, 3, 5])

    def test_diagonal(self):
        q = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * pq.m
        self.assertQuantityEqual(q.diagonal(offset=1), [2, 3] * pq.m)

    def test_compress(self):
        self.assertQuantityEqual(
            self.q.compress([False, True], axis=0),
            [[3, 4]] * pq.m
            )
        self.assertQuantityEqual(
            self.q.compress([False, True], axis=1),
            [[2], [4]] * pq.m
            )

    def test_searchsorted(self):
        self.assertQuantityEqual(
            self.q.flatten().searchsorted([1.5, 2.5] * pq.m),
            [1, 2]
            )

        self.assertRaises(ValueError, self.q.flatten().searchsorted, [1.5, 2.5])

    def test_nonzero(self):
        q = [1, 0, 5, 6, 0, 9] * pq.m
        self.assertQuantityEqual(q.nonzero()[0], [0, 2, 3, 5])

    def test_max(self):
        self.assertQuantityEqual(self.q.max(), 4*pq.m)

    def test_argmax(self):
        self.assertEqual(self.q.argmax(), 3)

    def test_min(self):
        self.assertEqual(self.q.min(), 1 * pq.m)

    def test_argmin(self):
        self.assertEqual(self.q.argmin(), 0)

    def test_ptp(self):
        self.assertQuantityEqual(self.q.ptp(), 3 * pq.m)

    def test_clip(self):
        self.assertQuantityEqual(
            self.q.copy().clip(max=2*pq.m),
            [[1, 2], [2, 2]] * pq.m
        )
        self.assertQuantityEqual(
            self.q.copy().clip(min=3*pq.m),
            [[3, 3], [3, 4]] * pq.m
        )
        self.assertQuantityEqual(
            self.q.copy().clip(min=2*pq.m, max=3*pq.m),
            [[2, 2], [3, 3]] * pq.m
        )
        self.assertRaises(ValueError, self.q.clip, pq.J)
        self.assertRaises(ValueError, self.q.clip, 1)

    def test_round(self):
        q = [1, 1.33, 5.67, 22] * pq.m
        self.assertQuantityEqual(q.round(0), [1, 1, 6, 22] * pq.m)
        self.assertQuantityEqual(q.round(-1), [0, 0, 10, 20] * pq.m)
        self.assertQuantityEqual(q.round(1), [1, 1.3, 5.7, 22] * pq.m)

    def test_trace(self):
        self.assertQuantityEqual(self.q.trace(), (1+4) * pq.m)

    def test_cumsum(self):
        self.assertQuantityEqual(self.q.cumsum(), [1, 3, 6, 10] * pq.m)

    def test_mean(self):
        self.assertQuantityEqual(self.q.mean(), 2.5 * pq.m)

    def test_var(self):
        self.assertQuantityEqual(self.q.var(), 1.25*pq.m**2)

    def test_std(self):
        self.assertQuantityEqual(self.q.std(), 1.11803*pq.m, delta=1e-5)

    def test_prod(self):
        self.assertQuantityEqual(self.q.prod(), 24 * pq.m**4)

    def test_cumprod(self):
        self.assertRaises(ValueError, self.q.cumprod)
        self.assertQuantityEqual((self.q/pq.m).cumprod(), [1, 2, 6, 24])

    def test_conj(self):
        self.assertQuantityEqual((self.q*(1+1j)).conj(), self.q*(1-1j))
        self.assertQuantityEqual((self.q*(1+1j)).conjugate(), self.q*(1-1j))

    def test_getitem(self):
        self.assertRaises(IndexError, self.q.__getitem__, (0,10))
        self.assertQuantityEqual(self.q[0], [1,2]*pq.m)
        self.assertQuantityEqual(self.q[1,1], 4*pq.m)

    def test_setitem (self):
        self.assertRaises(ValueError, self.q.__setitem__, (0,0), 1)
        self.assertRaises(ValueError, self.q.__setitem__, (0,0), 1*pq.J)
        self.assertRaises(ValueError, self.q.__setitem__, 0, 1)
        self.assertRaises(ValueError, self.q.__setitem__, 0, [1, 2])
        self.assertRaises(ValueError, self.q.__setitem__, 0, 1*pq.J)

        q = self.q.copy()
        q[0] = 1*pq.m
        self.assertQuantityEqual(q, [[1,1],[3,4]]*pq.m)

        q[0] = (1,2)*pq.m
        self.assertQuantityEqual(q, self.q)

        q[:] = 1*pq.m
        self.assertQuantityEqual(q, [[1,1],[1,1]]*pq.m)

        # check and see that dimensionless numbers work correctly
        q = [0,1,2,3]*pq.dimensionless
        q[0] = 1
        self.assertQuantityEqual(q, [1,1,2,3])
        q[0] = pq.m/pq.mm
        self.assertQuantityEqual(q, [1000, 1,2,3])

        q = [0,1,2,3] * pq.m/pq.mm
        q[0] = 1
        self.assertQuantityEqual(q, [0.001,1,2,3]*pq.m/pq.mm)

    def test_iterator(self):
        for q in self.q.flatten():
            self.assertQuantityEqual(q.units, pq.m)
