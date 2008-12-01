#+
#
# This file is part of h5py, a low-level Python interface to the HDF5 library.
#
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
#
# $Date$
#
#-

import unittest
import sys
import quantities
import test_quantities

TEST_CASES = (test_quantities.TestQuantities, )

def buildsuite(cases):

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    return suite

def runtests():
    suite = buildsuite(TEST_CASES)
    retval = unittest.TextTestRunner(verbosity=3).run(suite)
    print "=== Tested quantities %s ===" % (quantities.__version__)
    return retval.wasSuccessful()

def autotest():
    if not runtests():
        sys.exit(1)



