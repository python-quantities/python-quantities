from .. import units as pq
from .common import TestCase


class TestFormatting(TestCase):

    @staticmethod
    def _check(quantity, formatted):
        assert str(quantity) == formatted
        assert '{}'.format(quantity) == formatted
        assert '{!s}'.format(quantity) == formatted

    def test_str_format_scalar(self):
        self._check(1*pq.J, '1.0 J')

    def test_str_format_non_scalar(self):
        self._check([1, 2]*pq.J, '[1. 2.] J')
