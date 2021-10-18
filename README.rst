==========
quantities
==========

|pypi version| |pypi download| |Build status|

.. |pypi version| image:: https://img.shields.io/pypi/v/quantities.png
   :target: https://pypi.python.org/pypi/quantities
.. |Build status| image:: https://secure.travis-ci.org/python-quantities/python-quantities.png?branch=master
    :target: http://travis-ci.org/python-quantities/python-quantities

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

    $ python setup.py install --user

If you install it system-wide, you may need to prefix the previous command with ``sudo``::

    $ sudo python setup.py install

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
quantities is written by Darren Dale

License
-------
Quantities only uses BSD compatible code.  See the Open Source
Initiative `licenses page <http://www.opensource.org/licenses>`_
for details on individual licenses.

See `doc/user/license.rst <doc/user/license.rst>`_ for further details on the license of quantities
