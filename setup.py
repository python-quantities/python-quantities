from __future__ import with_statement

from distutils.cmd import Command
from distutils.errors import DistutilsError, DistutilsExecError
try:
    from setuptools import setup, Extension
except ImportError:
    from numpy.distutils.core import setup, Extension

import numpy

import os
import string
import sys

class test(Command):
    description = "Build quantities and run unit tests"
    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        buildobj = self.distribution.get_command_obj('build')
        buildobj.run()
        oldpath = sys.path
        try:
            sys.path = [os.path.abspath(buildobj.build_lib)] + oldpath
            import quantities.tests
            if not quantities.tests.runtests():
                raise DistutilsError("Unit tests failed.")
        finally:
            sys.path = oldpath

udunits = Extension('quantities.udunits',
                    ['udunits/src/udunits_py.c',
                     'udunits/src/utparse.c',
                     'udunits/src/utlib.c',
                     'udunits/src/utscan.c'],
                    include_dirs = ['udunits/include', numpy.get_include()])

with file('quantities/quantities-data/NIST_codata.txt') as f:
    data = f.read()
data = data.split('\n')[10:-1]

with file('quantities/constants/codata.py', 'w') as f:
    f.write('physical_constants = {}\n\n')
    for line in data:
        name = line[:55].rstrip().replace('mag.','magnetic').replace('mom.', 'moment')
        val = line[55:77].replace(' ','').replace('...','')
        prec = line[77:99].replace(' ','').replace('(exact)', '0')
        unit = line[99:].rstrip().replace(' ', '*').replace('^', '**')
        d = "{'value': %s, 'precision': %s, 'units': '%s'}"%(val, prec, unit)
        f.write("physical_constants['%s'] = %s\n"%(name, d))

desc = 'Support for physical quantities based on the popular numpy library'
long_desc = "Quantities is designed to handle arithmetic and conversions of \
physical quantities, which have a magnitude, dimensionality specified by \
various units, and possibly an uncertainty. Quantities is based on the popular \
numpy library. It is undergoing active development, and while the current \
features and API are fairly stable, test coverage is far from complete and the \
package is not ready for production use."
classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Education',
    'Topic :: Scientific/Engineering',
]


setup (name = "quantities",
       version = '0.1',
       author = 'Darren Dale',
       author_email = 'dsdale24@gmail.com',
       description = desc,
       keywords = ['quantities', 'physical quantities', 'units'],
       license = 'BSD',
       long_description = long_desc,
       classifiers = classifiers,
       platforms = 'Any',
       requires = ['numpy'],
       url = "http://packages.python.org/quantities",
       packages = ['quantities',
                   'quantities.units',
                   'quantities.constants',
                   'quantities.tests'],
       package_data = {'': ['quantities-data/udunits.dat']},
       ext_modules = [ udunits ],
       test_suite = 'nose.collector',
      )
