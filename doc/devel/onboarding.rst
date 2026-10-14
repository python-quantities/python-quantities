.. _onboarding:

***********
Onboarding
***********

This page is the starting point for new contributors. It walks through
setting up a development environment, running the tests, getting a high-level
sense of the codebase, and submitting a first pull request.

For deeper material, see also:

* :ref:`architecture-overview` — internal design of the package.
* :ref:`development-workflow` — branching, review, versioning and deprecation
  policy.
* :ref:`maintenance` — routine maintenance tasks for new committers.


Setting up a development environment
====================================

Fork the repository on GitHub (use the *Fork* button at the top of
https://github.com/python-quantities/python-quantities), then clone your
fork::

  git clone git@github.com:<your-github-username>/python-quantities.git
  cd python-quantities

It is also convenient to add the upstream repository as a remote, so you
can keep your fork in sync::

  git remote add upstream https://github.com/python-quantities/python-quantities.git

Create a virtual environment with a supported Python (see pyproject.toml). For
example, using the standard library ``venv`` module::

  python -m venv env
  source env/bin/activate

Install *quantities* in editable mode together with the test and
documentation extras::

  pip install -e ".[test,doc]"

The ``test`` extra pulls in `pytest`_ and `Ruff`_; ``doc`` pulls in
`Sphinx`_.

.. _pytest: https://docs.pytest.org/
.. _Sphinx: https://www.sphinx-doc.org/
.. _Ruff: https://docs.astral.sh/ruff/


Running the tests
=================

The full test suite is run with ``pytest`` from the repository root::

  pytest

To run just one test module, or one test::

  pytest quantities/tests/test_arithmetic.py
  pytest quantities/tests/test_arithmetic.py::TestQuantityArithmetic::test_add

The doctests in ``README.rst`` are also part of the suite that runs in CI::

  python -m doctest README.rst

If you have multiple NumPy versions available you can install a specific
version to verify behaviour against it.


Code map
========

A brief tour of the source tree (see :ref:`architecture-overview` for the
design rationale):

``quantities/quantity.py``
    Defines :class:`Quantity`, the main user-facing array type. Subclasses
    :class:`numpy.ndarray` and carries a ``_dimensionality`` attribute.

``quantities/unitquantity.py``
    Defines :class:`UnitQuantity` (a scalar :class:`Quantity` of magnitude 1
    representing a named unit) and its SI-base-dimension subclasses
    (:class:`UnitLength`, :class:`UnitMass`, :class:`UnitTime`, …).

``quantities/dimensionality.py``
    Implements :class:`Dimensionality`, a ``dict`` subclass that maps
    :class:`UnitQuantity` objects to integer exponents.

``quantities/registry.py``
    The ``unit_registry`` singleton and the safe-AST parser used to evaluate
    unit-expression strings such as ``"kg*m/s**2"`` without arbitrary code
    execution.

``quantities/umath.py``
    Wrappers around NumPy ufuncs (``sin``, ``cumsum``, ``gradient``,
    ``trapz``, …) that handle dimensional analysis correctly.

``quantities/uncertainquantity.py``
    :class:`UncertainQuantity`, a :class:`Quantity` subclass that propagates
    a ``_uncertainty`` attribute through arithmetic.

``quantities/units/``
    One module per physical dimension (length, mass, time, …). Each
    module defines :class:`UnitQuantity` instances that self-register on
    creation.

``quantities/constants/``
    Physical constants from the NIST CODATA dataset (``NIST_codata.txt``);
    the generated ``_codata.py`` module is rebuilt by ``python setup.py
    data``.

``quantities/tests/``
    The pytest suite.


Submitting a first pull request
===============================

1. Fork the repository on GitHub and clone your fork.
2. Create a topic branch off ``master``::

     git checkout -b my-fix

3. Make your changes. Please add or update tests so that the change is
   covered, and follow the existing docstring style (NumPy convention — see
   :ref:`documenting-quantities`).
4. Run the test suite (``pytest``) and the linter
   (``ruff check quantities``) locally.
5. Commit, push to your fork, and open a pull request against
   ``python-quantities/python-quantities:master``.

A maintainer will review the PR. Once CI passes and the review is approved,
the change will be merged.

For details on the review process, versioning and the deprecation policy,
see :ref:`development-workflow`.


Asking for help
===============

If anything in this guide is unclear, or you get stuck, please open a
GitHub issue or comment on a relevant existing one.
