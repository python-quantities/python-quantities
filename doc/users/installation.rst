************
Installation
************


Prerequisites
=============

Quantities has a few dependencies:

* Python_ (version 2.5 or later)
* NumPy_ (version 1.1 or later)

The following are also recommended:

* setuptools_ (version 0.6c8 or later)
* nose_ (for unit tests)

On several distributions, like Ubuntu, you may need to install the developer
tools and -dev libraries in order to use XPaXS.

Linux and OSX
=============

To install XPaXS on linux and OSX, download the XPaXS sourcecode from PyPi_
and run "python setup.py install" in the xpaxs source directory.

Windows
=======

A 32-bit windows installer is expected to be available at PyPi_ in 
January 2009. This will require the 32-bit python installation, even on 
a 64-bit machine, since most 3rd-party modules are compiled for the 
32-bit platform. 

Development Branch
==================

You can follow and contribute to XPaXS' development by obtaining a bzr version
control branch. Just install bzr and type::

  bzr branch lp:xpaxs

and then periodically bring your branch up to date::

  bzr pull

Bugs, feature requests, and questions can be directed to the launchpad_
website.


.. _Python: http://www.python.org/
.. _setuptools: http://peak.telecommunity.com/DevCenter/setuptools
.. _NumPy: http://www.scipy.org
.. _Nose: http://somethingaboutorange.com/mrl/projects/nose
.. _PyPi: http://pypi.python.org/pypi/xpaxs
.. _launchpad: https://launchpad.net/xpaxs
