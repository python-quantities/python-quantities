
import unittest
import tempfile
import shutil
import os
import numpy
import quantities as q

def test():
    assert 1==1, 'assert 1==1'

class TestQuantities(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(str(q.m), "m", str(q.m))
        self.assertEqual(str(q.J), "J", str(q.J))
        self.assertEqual(str(q.Quantity(1.0, q.J)), "1.0*(J)",
                         str(q.Quantity(1.0, q.J)))
        self.assertEqual(str(q.Quantity(1.0, 'J')), "1.0*(J)",
                         str(q.Quantity(1.0, 'J')))
        self.assertEqual(str(q.Quantity(1.0, 'kg*m**2/s**2')),
                         "1.0*(kg*m**2/s**2)",
                         str(q.Quantity(1.0, 'kg*m**2/s**2')))


