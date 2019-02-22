********
Releases
********

Creating Source Releases
========================

Quantities is distributed as a source release for Linux and Mac OS. To create a
source release, just do::

  python setup.py register
  python setup.py sdist --formats=zip,gztar
  twine upload dist/quantities-<x.y.z>.*

(replacing `x`, `y` and `z` appropriately).
This will create the tgz source file and upload it to the Python Package Index.
Uploading to PyPi requires a .pypirc file in your home directory, something
like::

  [server-login]
  username: <username>
  password: <password>

You can create a source distribution without uploading by doing::

  python setup.py sdist

This creates a source distribution in the `dist/` directory.


Creating Windows Installers
===========================

We distribute binary installers for the windows platform. In order to build the
windows installer, open a DOS window, cd into the quantities source directory
and run::

  python setup.py build
  python setup.py bdist_msi

This creates the executable windows installer in the `dist/` directory.


Building Quantities documentation
=================================

The Quantities documentation is automatically built on readthedocs.io.

Should you need to build the documentation locally,
Sphinx_, LaTeX_ (preferably `TeX-Live`_), and dvipng_ are
required. Once these are installed, do::

  cd doc
  make html

which will produce the html output and save it in build/sphinx/html. Then run::

  make latex
  cd build/latex
  make all-pdf
  cp Quantities.pdf ../html

which will generate a pdf file in the latex directory.

.. _Sphinx: http://sphinx.pocoo.org/
.. _LaTeX: http://www.latex-project.org/
.. _`TeX-Live`: http://www.tug.org/texlive/
.. _dvipng: http://savannah.nongnu.org/projects/dvipng/
