import warnings
from .. import QuantitiesDeprecationWarning, units as pq
from .common import TestCase
import numpy as np

class TestNumpy2_0(TestCase):


    # issues between NumPy < 2.0 and > 2.0 appear to be missed in current quantities testing
    # https://github.com/NeuralEnsemble/python-neo/pull/1490
    # this is a small test class to add changes to ensure pre and post 2.0 compatability
    
    def setUp(self):
        self.q = [[1, 2], [3, 4]] * pq.m

    def test_numpy_concatenate(self):

        concatenated_array = np.concatenate((self.q.flatten(), self.q.flatten()))
        self.assertQuantityEqual(concatenated_array, [1,2,3,4]*pq.m)
    
    def test_numpy_dtype(self):

        test_array_float32 = np.arange(10, dtype=np.float32) * pq.s
        self.assertEqual(test_array_float32.dtype, np.float32)