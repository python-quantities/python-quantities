# -*- coding: utf-8 -*-

from .. import units as pq
from .common import TestCase
import numpy as np

class TestQuantityMethods(TestCase):

    def setUp(self):
        self.q = [[1, 2], [3, 4]] * pq.m

    def test_tolist(self):
        self.assertEqual(self.q.tolist(), [[1*pq.m, 2*pq.m], [3*pq.m, 4*pq.m]])

    def test_sum(self):
        self.assertQuantityEqual(self.q.sum(), 10*pq.m)
        self.assertQuantityEqual(self.q.sum(0), [4, 6]*pq.m)
        self.assertQuantityEqual(self.q.sum(1), [3, 7]*pq.m)

    def test_nansum(self):
        import numpy as np
        qnan = [[1,2], [3,4], [np.nan,np.nan]] * pq.m
        self.assertQuantityEqual(qnan.nansum(), 10*pq.m )
        self.assertQuantityEqual(qnan.nansum(0), [4,6]*pq.m )

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

    def methodWithOut(self, name, result, q=None, *args, **kw):
        import numpy as np
        from .. import Quantity

        if q is None:
            q = self.q

        self.assertQuantityEqual(
            getattr(q.copy(), name)(*args,**kw),
            result
        )
        if isinstance(result, Quantity):
            # deliberately using an incompatible unit
            out = Quantity(np.empty_like(result.magnitude), pq.s, copy=False)
        else:
            out = np.empty_like(result)
        ret = getattr(q.copy(), name)(*args, out=out, **kw)
        self.assertQuantityEqual(
            ret,
            result
        )
        # returned array should be the same as out
        self.assertEqual(id(ret),id(out))
        # but the units had to be adjusted
        if isinstance(result, Quantity):
            self.assertEqual(ret.units,result.units)
        else:
            self.assertEqual(
                getattr(ret, 'units', pq.dimensionless),
                pq.dimensionless
            )


    def test_max(self):
        self.methodWithOut('max', 4 * pq.m)
        self.methodWithOut('max', [3, 4] * pq.m, axis=0)
        self.methodWithOut('max', [2, 4] * pq.m, axis=1)

    def test_nanmax(self):
        q = np.append(self.q, np.nan) * self.q.units
        self.assertQuantityEqual(q.nanmax(), 4*pq.m)

    def test_argmax(self):
        import numpy as np
        self.assertQuantityEqual(self.q.argmax(), 3)
        self.assertQuantityEqual(self.q.argmax(axis=0), [1, 1])
        self.assertQuantityEqual(self.q.argmax(axis=1), [1, 1])
        # apparently, numpy's argmax does not return the same object when out is specified.
        # instead, we test here for shared data
        out = np.r_[0, 0]
        ret = self.q.argmax(axis=0,out=out)
        self.assertQuantityEqual(ret, [1, 1])
        self.assertEqual(ret.ctypes.data, out.ctypes.data)

    def test_nanargmax(self):
        q = np.append(self.q, np.nan) * self.q.units
        self.assertEqual(self.q.nanargmax(), 3)

    def test_min(self):
        self.methodWithOut('min', 1 * pq.m)
        self.methodWithOut('min', [1, 2] * pq.m, axis=0)
        self.methodWithOut('min', [1, 3] * pq.m, axis=1)

    def test_nanmin(self):
        q = np.append(self.q, np.nan) * self.q.units
        self.assertQuantityEqual(q.nanmin(), 1*pq.m)

    def test_argmin(self):
        import numpy as np
        self.assertQuantityEqual(self.q.argmin(), 0)
        self.assertQuantityEqual(self.q.argmin(axis=0), [0, 0])
        self.assertQuantityEqual(self.q.argmin(axis=1), [0, 0])
        # apparently, numpy's argmax does not return the same object when out is specified.
        # instead, we test here for shared data
        out = np.r_[2, 2]
        ret = self.q.argmin(axis=0,out=out)
        self.assertQuantityEqual(ret, [0, 0])
        self.assertEqual(ret.ctypes.data, out.ctypes.data)

    def test_nanargmax(self):
        q = np.append(self.q, np.nan) * self.q.units
        self.assertEqual(self.q.nanargmin(), 0)
        
    def test_ptp(self):
        self.methodWithOut('ptp', 3 * pq.m)
        self.methodWithOut('ptp', [2, 2] * pq.m, axis=0)
        self.methodWithOut('ptp', [1, 1] * pq.m, axis=1)

    def test_clip(self):
        self.methodWithOut(
            'clip',
            [[1, 2], [2, 2]] * pq.m,
            max=2*pq.m,
        )
        self.methodWithOut(
            'clip',
            [[3, 3], [3, 4]] * pq.m,
            min=3*pq.m,
        )
        self.methodWithOut(
            'clip',
            [[2, 2], [3, 3]] * pq.m,
            min=2*pq.m, max=3*pq.m
        )
        self.assertRaises(ValueError, self.q.clip, pq.J)
        self.assertRaises(ValueError, self.q.clip, 1)

    def test_round(self):
        q = [1, 1.33, 5.67, 22] * pq.m
        self.methodWithOut(
            'round',
            [1, 1, 6, 22] * pq.m,
            q=q,
            decimals=0,
        )
        self.methodWithOut(
            'round',
            [0, 0, 10, 20] * pq.m,
            q=q,
            decimals=-1,
        )
        self.methodWithOut(
            'round',
            [1, 1.3, 5.7, 22] * pq.m,
            q=q,
            decimals=1,
        )

    def test_trace(self):
        self.methodWithOut('trace', (1+4) * pq.m)

    def test_cumsum(self):
        self.methodWithOut('cumsum', [1, 3, 6, 10] * pq.m)
        self.methodWithOut('cumsum', [[1, 2], [4, 6]] * pq.m, axis=0)
        self.methodWithOut('cumsum', [[1, 3], [3, 7]] * pq.m, axis=1)

    def test_mean(self):
        self.methodWithOut('mean', 2.5 * pq.m)
        self.methodWithOut('mean', [2, 3] * pq.m, axis=0)
        self.methodWithOut('mean', [1.5, 3.5] * pq.m, axis=1)

    def test_nanmean(self):
        import numpy as np    
        q = [[1,2], [3,4], [np.nan,np.nan]] * pq.m
        self.assertQuantityEqual(q.nanmean(), self.q.mean())

    def test_var(self):
        self.methodWithOut('var', 1.25 * pq.m**2)
        self.methodWithOut('var', [1, 1] * pq.m**2, axis=0)
        self.methodWithOut('var', [0.25, 0.25] * pq.m**2, axis=1)

    def test_std(self):
        self.methodWithOut('std', 1.1180339887498949 * pq.m)
        self.methodWithOut('std', [1, 1] * pq.m, axis=0)
        self.methodWithOut('std', [0.5, 0.5] * pq.m, axis=1)

    def test_nanstd(self):
        import numpy as np    
        q0 = [[1,2], [3,4]] * pq.m
        q1 = [[1,2], [3,4], [np.nan,np.nan]] * pq.m
        self.assertQuantityEqual(q0.std(), q1.nanstd())

    def test_prod(self):
        self.methodWithOut('prod', 24 * pq.m**4)
        self.methodWithOut('prod', [3, 8] * pq.m**2, axis=0)
        self.methodWithOut('prod', [2, 12] * pq.m**2, axis=1)

    def test_cumprod(self):
        self.assertRaises(ValueError, self.q.cumprod)
        self.assertQuantityEqual((self.q/pq.m).cumprod(), [1, 2, 6, 24])
        q = self.q/pq.m
        self.methodWithOut(
            'cumprod',
            [1, 2, 6, 24],
            q=q,
        )
        self.methodWithOut(
            'cumprod',
            [[1, 2], [3, 8]],
            q=q,
            axis=0,
        )
        self.methodWithOut(
            'cumprod',
            [[1, 2], [3, 12]],
            q=q,
            axis=1,
        )

    def test_conj(self):
        self.assertQuantityEqual((self.q*(1+1j)).conj(), self.q*(1-1j))
        self.assertQuantityEqual((self.q*(1+1j)).conjugate(), self.q*(1-1j))

    def test_real(self):
        test_q = self.q * (1 + 1j)
        test_q.real = [[39.3701, 39.3701], [39.3701, 39.3701]] * pq.inch
        self.assertQuantityEqual(test_q.real, [[1., 1.], [1., 1.]] * pq.m)

    def test_imag(self):
        test_q = self.q * (1 + 1j)
        test_q.imag = [[39.3701, 39.3701], [39.3701, 39.3701]] * pq.inch
        self.assertQuantityEqual(test_q.imag, [[1., 1.], [1., 1.]] * pq.m)

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
