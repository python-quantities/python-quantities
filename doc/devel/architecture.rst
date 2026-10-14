.. _architecture-overview:

*********************
Architecture Overview
*********************

This page gives a high-level view of how *quantities* is put together
internally, intended for new contributors who want to understand the design
before making non-trivial changes.

For the user-facing API, see the :doc:`user guide </user/index>`.


Design rationale
================

*quantities* is a NumPy extension library. Rather than implement its own
array machinery, it subclasses :class:`numpy.ndarray` so that
:class:`Quantity` instances behave like regular arrays — they participate
in vectorised arithmetic, broadcasting, ufuncs and indexing — while also
carrying dimensional (unit) information that is validated and propagated
through every operation.

The design has three pillars:

1. **Dimensionality** is data, not a string. Units multiply, divide and
   raise to integer powers; *quantities* stores those exponents in a
   mapping and uses it both for arithmetic and for display.
2. **Conversions** are pure scalar multiplications. The unit registry
   produces conversion factors that are simply numbers; the dimensional
   bookkeeping lives in the :class:`Dimensionality` mapping.
3. **Interoperability with NumPy is paramount.** *quantities* hooks into
   the ufunc machinery so that ``np.sin``, ``np.sqrt``, ``np.add`` and so
   on work as expected on :class:`Quantity` arrays.


Class hierarchy
===============

The core type hierarchy is::

    numpy.ndarray
        Quantity                       # quantity.py
            UnitQuantity               # unitquantity.py
                UnitLength             # quantities/units/length.py
                UnitMass               # quantities/units/mass.py
                UnitTime               # quantities/units/time.py
                ...                    # one subclass per SI base dimension
                CompoundUnit           # preserves compound expressions
            UncertainQuantity          # uncertainquantity.py

:class:`Quantity`
    Array values plus a ``_dimensionality`` attribute. This is what users
    create with expressions such as ``42 * pq.metre``.

:class:`UnitQuantity`
    A scalar :class:`Quantity` with magnitude 1 that represents a named
    unit (``pq.metre``, ``pq.second``, …). Every :class:`UnitQuantity`
    self-registers in the global ``unit_registry`` on construction.

:class:`UnitLength`, :class:`UnitMass`, …
    SI-base-dimension subclasses of :class:`UnitQuantity`. They behave the
    same as :class:`UnitQuantity` but their type marks the base dimension
    they represent.

:class:`CompoundUnit`
    A :class:`UnitQuantity` that preserves the compound expression
    (e.g. ``m**2/m**3``) instead of simplifying it. Useful for display
    when the user wants the original notation kept.

:class:`UncertainQuantity`
    A :class:`Quantity` subclass that carries a ``_uncertainty`` attribute
    (itself a :class:`Quantity`) and propagates uncertainty through
    arithmetic.


Dimensionality
==============

:class:`Dimensionality` (``dimensionality.py``) is a :class:`dict`
subclass that maps :class:`UnitQuantity` objects to integer exponents.
For example, the dimensionality of a velocity is::

    {metre: 1, second: -1}

The class supports several text representations — ``.string``, ``.unicode``,
``.latex`` and ``.html`` — produced by ``markup.py``. Dimension checking
and conversion-factor computation happen in ``quantity.py``
(``get_conversion_factor``, ``validate_dimensionality``).


The unit registry
=================

:class:`UnitRegistry` (``registry.py``) is a singleton that maps string
names and symbols to :class:`UnitQuantity` objects. When the user writes::

    q.units = "kg*m/s**2"

the string is parsed by the registry. To avoid arbitrary code execution,
the parser uses Python's :mod:`ast` module and only allows a restricted
set of node types (names, numeric literals, binary arithmetic, power).
This is safer than ``eval`` and is verified by the test suite.

:class:`UnitQuantity` subclasses register themselves in this registry on
construction via their ``__init__``, so importing a module from
``quantities/units/`` is enough to make all the units it defines
available.


Math support
============

``umath.py`` wraps NumPy ufuncs and reductions (trigonometric functions,
``cumsum``, ``gradient``, ``trapz``, …) so they handle dimensional
analysis correctly. For example, ``pq.sin`` checks that the input has
units of angle, whereas plain ``np.sin`` silently extracts the magnitude.

The ``known issues`` section of the user guide documents which NumPy
operations are not yet dimension-aware.


The ``PREFERRED`` list
======================

``quantity.py`` defines a module-level list called ``PREFERRED`` which
downstream packages can mutate to express their preferred units for
display and simplification. This is the main extension point used by
domain-specific consumers of *quantities*.


Units and constants data
========================

* ``quantities/units/`` — one module per physical dimension. Each module
  defines :class:`UnitQuantity` instances which self-register.
* ``quantities/constants/`` — physical constants as :class:`UnitConstant`
  objects (a subclass of :class:`UncertainQuantity`). The values come
  from ``NIST_codata.txt``; ``_codata.py`` is auto-generated from that
  file by ``python setup.py data``.
