import os
import sys

import unittest

suite = unittest.TestLoader().discover('quantities')
unittest.TextTestRunner(verbosity=1).run(suite)
