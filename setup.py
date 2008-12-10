from __future__ import with_statement

from distutils.cmd import Command
from distutils.errors import DistutilsError, DistutilsExecError
try:
    from setuptools import setup, Extension
except ImportError:
    from numpy.distutils.core import setup, Extension

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
                    include_dirs = ['udunits/include'])

with file('quantities/quantities-data/NIST_codata.txt') as f:
    data = f.read()
data = data.split('\n')[10:-1]

with file('quantities/constants/codata.py', 'w') as f:
    f.write('physical_constants = {}\n\n')
    for line in data:
        name = line[:55].rstrip().replace('mag.','magnetic').replace('mom.', 'moment')
        val = line[55:77].replace(' ','').replace('...','')
        prec = line[77:99].replace(' ','').replace('(exact)', '0')
        unit = line[99:].rstrip().replace(' ', '*')
        unit = unit.replace('T', 'T_').replace('Pa','Pa_')
        unit = unit.replace('J','J_').replace('ohm','ohm_')
        unit = unit.replace('E_h','E_h_').replace('V','V_')
        unit = unit.replace('Hz','Hz_').replace('F','F_')
        unit = unit.replace('W','W_').replace('W_b','Wb_')
        d = "{'value': %s, 'precision': %s, 'units': '%s'}"%(val, prec, unit)
        f.write("physical_constants['%s'] = %s\n"%(name, d))


setup (name = "quantities",
       version='1.0',
       author='doutriaux1@llnl.gov',
       description = "Python wrapping for UDUNITS package developed by UNIDATA",
       url = "http://www-pcmdi.llnl.gov/software",
       packages = ['quantities',
                   'quantities.units',
                   'quantities.constants',
                   'quantities.tests'],
       package_data = {'': ['quantities-data/udunits.dat']},
       ext_modules = [ udunits ],
       test_suite = 'nose.collector',
      )
