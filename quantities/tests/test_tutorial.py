import doctest

import quantities

from .common import TestCase


class TestTutorial(TestCase):
    """The quick-start tutorial in the docs is the package docstring."""

    def test_tutorial_examples(self):
        result = doctest.testmod(
            quantities, optionflags=doctest.NORMALIZE_WHITESPACE, report=True
        )
        self.assertGreater(result.attempted, 0)
        self.assertEqual(result.failed, 0)
