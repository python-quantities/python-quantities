==========
quantities
==========

Quantities is designed to handle arithmetic and
conversions of physical quantities, which have a magnitude, dimensionality
specified by various units, and possibly an uncertainty. See the tutorial_
for examples. Quantities builds on the popular numpy library and is
designed to work with numpy ufuncs, many of which are already
supported. Quantities is actively developed, and while the current features
and API are stable, test coverage is incomplete so the package is not
suggested for mission-critical applications.

|pypi version|_ |Build status|_

.. |pypi version| image:: https://img.shields.io/pypi/v/quantities.png
.. _`pypi version`: https://pypi.python.org/pypi/quantities
.. |Build status| image:: https://github.com/python-quantities/python-quantities/actions/workflows/test.yml/badge.svg?branch=master
.. _`Build status`: https://github.com/python-quantities/python-quantities/actions/workflows/test.yml
.. _tutorial: http://python-quantities.readthedocs.io/en/latest/user/tutorial.html


A Python package for handling physical quantities. The source code and issue
tracker are hosted on GitHub:

https://www.github.com/python-quantities/python-quantities

Download
--------
Get the latest version of quantities from
https://pypi.python.org/pypi/quantities/

To get the Git version do::

    $ git clone git://github.com/python-quantities/python-quantities.git


Documentation and usage
-----------------------
You can find the official documentation at:

http://python-quantities.readthedocs.io/

Here is a simple example:

.. code:: python

   >>> import quantities as pq
   >>> distance = 42*pq.metre
   >>> time = 17*pq.second
   >>> velocity = distance / time
   >>> "%.3f %s" % (velocity.magnitude, velocity.dimensionality)
   '2.471 m/s'
   >>> velocity + 3
   Traceback (most recent call last):
     ...
   ValueError: Unable to convert between units of "dimensionless" and "m/s"

Installation
------------
quantities has a hard dependency on the `NumPy <http://www.numpy.org>`_ library.
You should install it first, please refer to the NumPy installation guide:

http://docs.scipy.org/doc/numpy/user/install.html

To install quantities itself, then simply run::

    $ pip install quantities


Tests
-----
To execute all tests, install pytest::

    $ python -m pip install pytest

And run::

    $ pytest

in the current directory. The master branch is automatically tested by
GitHub Actions.

Author
------
quantities was originally written by Darren Dale, and has received contributions from `many people`_.

.. _`many people`: https://github.com/python-quantities/python-quantities/graphs/contributors

License
-------
Quantities only uses BSD compatible code.  See the Open Source
Initiative `licenses page <http://www.opensource.org/licenses>`_
for details on individual licenses.

See `doc/user/license.rst <doc/user/license.rst>`_ for further details on the license of quantities
