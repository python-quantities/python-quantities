import sys
import unittest

import numpy as np

from ..quantity import Quantity
from ..units import set_default_units

class TestCase(unittest.TestCase):

    def setUp(self):
        set_default_units('SI')

    def tearDown(self):
        set_default_units('SI')

    def assertQuantityEqual(self, q1, q2, msg=None, delta=None):
        """
        Make sure q1 and q2 are the same quantities to within the given
        precision.
        """
        if delta is None:
            # NumPy 2 introduced float16, so we base tolerance on machine epsilon
            delta1 = np.finfo(q1.dtype).eps if isinstance(q1, np.ndarray) and q1.dtype.kind in 'fc' else 1e-15
            delta2 = np.finfo(q2.dtype).eps if isinstance(q2, np.ndarray) and q2.dtype.kind in 'fc' else 1e-15
            delta = max(delta1, delta2)**0.3
        msg = '' if msg is None else ' (%s)' % msg

        q1 = Quantity(q1)
        q2 = Quantity(q2)
        if q1.shape != q2.shape:
            raise self.failureException(
                f"Shape mismatch ({q1.shape} vs {q2.shape}){msg}"
                )

        if not np.all(np.abs(q1.magnitude - q2.magnitude) < delta):
            raise self.failureException(
                "Magnitudes differ by more than %g (%s vs %s)%s"
                % (delta, q1.magnitude, q2.magnitude, msg)
                )

        d1 = getattr(q1, '_dimensionality', None)
        d2 = getattr(q2, '_dimensionality', None)
        if (d1 or d2) and not (d1 == d2):
            raise self.failureException(
                f"Dimensionalities are not equal ({d1} vs {d2}){msg}"
                )
