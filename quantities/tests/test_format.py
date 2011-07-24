# -*- coding: utf-8 -*-

import unittest

import quantities as pq

_INT_TYPES = ('b', 'd', 'o', 'x', 'X')
_ANY_TYPES = ('n', 'e', 'E', 'f', 'F', 'g', 'G', '%')


class TestUnits(unittest.TestCase):

    def _test_ok(self, value, ntypes):
        q = pq.Quantity(value, 'm/s')
        for ntype in ntypes:
            spec = '{0:' + ntype + '}'
            self.assertEqual(spec.format(value), spec.format(q))

    def _test_raises(self, value, ntypes):
        q = pq.Quantity(value, 'm/s')
        for ntype in ntypes:
            spec = '{0:' + ntype + '}'
            self.assertRaises(ValueError, spec.format, q)

    def test_format_int(self):
        """
        Formatting to an int format spec
        """
        self._test_ok(1, _INT_TYPES)
        self._test_ok(1, _ANY_TYPES)

    def test_format_float(self):
        """
        Formatting a float quantity to a generic format spec
        """
        self._test_ok(1.3, _ANY_TYPES)

    def test_format_float_as_int(self):
        """
        Formatting a float quantity to an int format spec should fail
        """
        self._test_raises(1.3, _INT_TYPES)

if __name__ == "__main__":
    unittest.main()
