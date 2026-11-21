=======
Changes
=======

----------
Unreleased
----------

- Support ``np.sign`` on quantities, returning a dimensionless result
  (`#114 <https://github.com/python-quantities/python-quantities/issues/114>`_).

------
0.17.0
------

- Drop support for Python 3.10 and NumPy 1.x, following NEP 29 plus one year, and add CI testing for Python 3.15 and NumPy 2.5 (`PR#277 <https://github.com/python-quantities/python-quantities/pull/277>`_)
- **Behaviour change:** a NumPy ufunc whose ``out`` argument is a view of a Quantity now raises ``ValueError`` if the result would change the units of the view, as in-place operators already did. Values already written to ``out`` are not rolled back (`PR#276 <https://github.com/python-quantities/python-quantities/pull/276>`_)
- Fix ``np.nanstd``, and the units of ufunc results written to an existing Quantity with ``out``, with NumPy 2 (`#223 <https://github.com/python-quantities/python-quantities/issues/223>`_, `PR#276 <https://github.com/python-quantities/python-quantities/pull/276>`_)
- Add lumen, lux, foot-candle, phot, nit and stilb photometric units (`#286 <https://github.com/python-quantities/python-quantities/issues/286>`_, `PR#288 <https://github.com/python-quantities/python-quantities/pull/288>`_)
- Support unary ``+`` (``np.positive``) on quantities (`#94 <https://github.com/python-quantities/python-quantities/issues/94>`_, `PR#273 <https://github.com/python-quantities/python-quantities/pull/273>`_)
- Correct the pond force unit to 0.00980665 N (one gram-force); it was previously 1000 times too large (`#44 <https://github.com/python-quantities/python-quantities/issues/44>`_, `PR#275 <https://github.com/python-quantities/python-quantities/pull/275>`_)
- Fix ``np.ptp`` dropping the units of a Quantity with NumPy 2 (`PR#281 <https://github.com/python-quantities/python-quantities/pull/281>`_)
- Allow numeric bounds when clipping dimensionless quantities (`#108 <https://github.com/python-quantities/python-quantities/issues/108>`_, `PR#282 <https://github.com/python-quantities/python-quantities/pull/282>`_)
- Convert quantity-valued ``ediff1d`` boundary differences (``to_begin``, ``to_end``) to the input units (`PR#283 <https://github.com/python-quantities/python-quantities/pull/283>`_)
- ``rescale()`` to the same units now returns a new Quantity built in the same way as for different units, so ``.base`` is no longer ``None`` (`#42 <https://github.com/python-quantities/python-quantities/issues/42>`_)
- Fix constructing an ``UncertainQuantity`` from another ``UncertainQuantity`` without an explicit uncertainty, which raised ``AttributeError`` (`PR#289 <https://github.com/python-quantities/python-quantities/pull/289>`_)
- Adding or subtracting a number or Quantity without an uncertainty to or from an ``UncertainQuantity`` now treats it as exact, instead of raising ``AttributeError`` (`#45 <https://github.com/python-quantities/python-quantities/issues/45>`_)
- ``Quantity.nanmax()``, ``nanmin()``, ``nanargmax()`` and ``nanargmin()`` now honour ``axis`` and ``out``; they used to reduce over the whole array and ignore ``out`` (`PR#294 <https://github.com/python-quantities/python-quantities/pull/294>`_)
- ``quantities.umath.trapz`` now always uses ``numpy.trapezoid``; SciPy is no longer used, and the ``scipy`` optional dependency has been removed (`PR#285 <https://github.com/python-quantities/python-quantities/pull/285>`_)
- Functions in ``quantities.umath`` no longer emit a spurious ``QuantitiesDeprecationWarning`` about the ``copy`` argument (`PR#285 <https://github.com/python-quantities/python-quantities/pull/285>`_)
- Fix the import of ``__version__`` when the package is built with setuptools_scm older than 7.0, which is now the minimum required (`#224 <https://github.com/python-quantities/python-quantities/issues/224>`_)
- Fix type checking with NumPy 2.5 (`PR#277 <https://github.com/python-quantities/python-quantities/pull/277>`_)
- Documentation: the tutorial is updated for Python 3, covers ``.magnitude`` and ``.units``, and is now tested; the documentation now says which operations propagate the uncertainty of an ``UncertainQuantity`` (`#214 <https://github.com/python-quantities/python-quantities/issues/214>`_, `#46 <https://github.com/python-quantities/python-quantities/issues/46>`_, `PR#291 <https://github.com/python-quantities/python-quantities/pull/291>`_)
- Development: extended documentation for developers and maintainers (`PR#274 <https://github.com/python-quantities/python-quantities/pull/274>`_); tests for ``quantities.constants`` (`PR#280 <https://github.com/python-quantities/python-quantities/pull/280>`_); the generated constants file records the CODATA edition instead of a timestamp (`PR#279 <https://github.com/python-quantities/python-quantities/pull/279>`_); f-strings throughout (`PR#230 <https://github.com/python-quantities/python-quantities/pull/230>`_); Ruff linting, now enforced in CI (`PR#290 <https://github.com/python-quantities/python-quantities/pull/290>`_)

------
0.16.4
------

- Fix problems with v0.16.3: deepcopy was not copying all necessary attributes of UnitQuantity; in-place arithmetic didn't handle multiplying by scalars

------
0.16.3
------

- Fix broken `deepcopy` behaviour when using quantities with NumPy 2.x (`PR#257 <https://github.com/python-quantities/python-quantities/pull/257>`_)
- Fixes for assorted breakages that appear with recent NumPy versions (`PR#262 <https://github.com/python-quantities/python-quantities/pull/262>`_, `PR#267 <https://github.com/python-quantities/python-quantities/pull/267>`_)
- Revert to previous behaviour where spaces around units were allowed  (`PR#270 <https://github.com/python-quantities/python-quantities/pull/270>`_)
- Fix broken in-place arithmetic when using quantities with NumPy 2.x (`PR#271 <https://github.com/python-quantities/python-quantities/pull/271>`_)
- Add CI testing for Python 3.13 and 3.14, drop support for Python 3.9
- Add CI testing for NumPy versions 2.1 to 2.4, drop support for NumPy 1.22 and 1.23.

------
0.16.2
------

- Added a property `dimensionless_magnitude` to the Quantity class ([`PR#248 <https://github.com/python-quantities/python-quantities/pull/248>`_])
- Implemented an alternative approach to avoiding arbitrary evaluation of code when parsing strings as units, which fixes some bugs introduced in v0.16.1 ([`PR#251 <https://github.com/python-quantities/python-quantities/pull/251>`_])
- Added the kilonewton

------
0.16.1
------

- Fixed a couple of small bugs ([`PR#238 <https://github.com/python-quantities/python-quantities/pull/238>`_] and [`PR#242 <https://github.com/python-quantities/python-quantities/pull/242>`_])
- Added umath funcs: `maximum` & `minimum`

------
0.16.0
------

- Added support for NumPy 2.0, while maintaining support for older versions back to 1.22 [`PR#235 <https://github.com/python-quantities/python-quantities/pull/235>`_]. Many thanks to Björn Dahlgren and Zach McKenzie for this.
- Fixed a potential security hole [`PR#236 <https://github.com/python-quantities/python-quantities/pull/236>`_]
- Dropped support for Python 3.8

------
0.15.0
------

- Quantities now has type stubs for all classes and functions. Many thanks to Peter Konradi and Takumasa Nakamura for this major effort.
- Fixed a number of deprecations coming from NumPy (thanks to Zach McKenzie)
- Dropped support for NumPy 1.19, added testing for Numpy 1.25 and 1.26, and for Python 3.12

------
0.14.1
------

- Fixed a bug when scaling quantities with integer dtype [`PR#216 <https://github.com/python-quantities/python-quantities/pull/216>`_]

------
0.14.0
------

- Added decimeter to list of supported units [`PR#202 <https://github.com/python-quantities/python-quantities/pull/202>`_]
- Removed deprecated mb as symbol for millibar unit [`PR#203 <https://github.com/python-quantities/python-quantities/pull/203>`_]
- Fixed failure to preserve dtype in rescale [`PR#204 <https://github.com/python-quantities/python-quantities/pull/204>`_]
- Added exp2 as a supported ufunc
- Fixed failure to handle units with floordiv [`PR#207 <https://github.com/python-quantities/python-quantities/pull/207>`_]
- Added femtofarad (fF) to list of supported units
- Dropped support for Python 3.7
- Dropped support for NumPy versions older than 1.19
- Converted the project packaging from setup.py-based to pyproject.toml-based


------
0.13.0
------

- Dropped support for Python versions older than 3.7, in particular, for Python 2.7.
- Dropped support for NumPy versions older than 1.16
- Switched test runner to pytest, and CI to Github Actions


------
0.12.5
------

- Added preferred units support for .rescale
- Added dimensionless unit 'lsb' (least significant bit)
- Added SI multiples for Kelvin
- Fixed invalid escape sequence

All changes
***********

https://github.com/python-quantities/python-quantities/issues?utf8=✓&q=is%3Aclosed+closed%3A2020-01-08..2021-08-16

------
0.12.4
------

- Fix broken support for `pq.Quanitty('mbar')`
- Add a `__format__` implementation for Quantity
- Fix `np.arctan2` regression due to newer numpy version
- Fix "<ufunc 'clip'> not supported" error
- Test against Python 3.8 and NumPy 1.17

All changes
***********

https://github.com/python-quantities/python-quantities/issues?utf8=✓&q=is%3Aclosed+closed%3A2019-02-23..2020-01-08+

------
0.12.3
------

Updates to support NumPy up to version 1.16.1, and Python 3.7.
Added microcoulomb and millicoulomb units.

All changes
***********

https://github.com/python-quantities/python-quantities/issues?utf8=✓&q=is%3Aclosed%20closed%3A2018-07-03..2019-02-22


------
0.12.2
------

Added SI multiples for the byte unit (kB, MB, ...) and the IEC units (KiB, MiB...).

All changes
***********

https://github.com/python-quantities/python-quantities/issues?utf8=✓&q=is%3Aclosed%20closed%3A2017-09-01..2018-07-02


------
0.12.1
------

Bugs fixed
**********

https://github.com/python-quantities/python-quantities/issues?utf8=✓&q=is%3Aclosed%20closed%3A2017-08-02..2017-08-30

------
0.12.0
------

Removed support for Python 2.6, since NumPy removed support for it as of
version 1.12. Numpy-1.8.2 or later is now required.

Added more ufuncs: equal, not_equal, less, less_equal, greater, greater_equal


Bugs fixed
**********

https://github.com/python-quantities/python-quantities/issues?utf8=✓&q=is%3Aissue%20is%3Aclosed%20closed%3A2015-12-06..2017-08-01


------
0.11.0
------

Added many new unit definitions, including aliases for American/British spellings of liter/litre
The Quantity class can now be subclassed.
Supports `np.fabs`
The test suite is now run with Travis CI


Bugs fixed
**********

https://github.com/python-quantities/python-quantities/issues?utf8=✓&q=is%3Aissue%20is%3Aclosed%20closed%3A2011-09-27..2015-12-06


------
0.10.0
------

The dalton has been added as an alias for the unified atomic mass
unit, in response to the 2005 IUPAP report recognizing that both
terms should be officially recognized. Da is recognized as the symbol
for the dalton.

The test suite has been refactored to use the unittest package
provided by the python-2.7 and python-3.2 standard library. With
python-2.6 or python-3.{0,1}, unittest2 is required. The test suite
is run with::

   python setup.py test

Quantities no longer requires or uses Distribute or Setuptools.


The test framework has been refactored to use unittest exclusively
(requires unittest2 with  python-2.6, python-3.0, and python-3.1).
The test suite is now run in the source directory::

   python setup.py test

Added a defintion for the centipoise.

Bugs fixed
**********

* #11 incorrect definition for millidarcy
* #12 add definition of centipoise
* #13 add ddof keyword for std, var methods
* #16 add "latex" property to dimensionality for latex format strings
* #19 Astronomical Unit definition updated to reflect 2009 spec


-----
0.9.0
-----

Numpy-1.4.0 or later is also required, in order to properly handle
units during in-place arithmetic.

Physical constants are now instances of Quantity, rather than
UncertainQuantity. Development on UncertainQuantity is not as
advanced, so hopefully this change will make the constants package
more robust.

Bugs fixed
**********

* #1 use revolution/min instead of 1/min as definition of rpm
* #2 silently fail to import test runner if nose is not installed
* #4 remove the "jiffy", as there are conflicting definitions
  depending on (or even within a) discipline.
* #5 fix the definition of femtometer

-----
0.8.0
-----

Quantities now requires Python-2.6 or later. Python-3 is supported.

The fill method, and __setitem__ methods will now accept scalar
values in addition to quantities. If a scalar is passed, an
assumption is made that the change should only be applied to the
quantity's magnitude. The fill method will also accept a quantity
with incompatible units, and will overwrite the previous units.

The log and exp functions have been removed. Quantities will work
with numpy's version of these functions.

Quantities development has migrated from bzr/launchpad to
git/github. Please report problems to
http://github.com/python-quantities/python-quantities or visit the
mailing list at http://groups.google.com/group/python-quantities

Bugs fixed
**********

* #501563 incorrect conversion ratio for gills/pints
* #515314 don't memoize simplified, fixes set_default_units
* #526008 add definition of stone, used in UK to measure mass
* #525991 support specifying units using unicode
* #529266 fix conversion to gallons

-----
0.7.0
-----

A critical problem was identified, caused by a conflict between farad
and Faraday identified in quantities <0.7. F is commonly used as a
symbol for both the farad and Faraday's constant. This is problematic
for quanitities, which registers its units so they can be accessed
using a string corresponding to the unit name or symbol. In this
case, the Faraday constant was registered after the farad, so any
quantity that was expressed in terms of farads would be incorrectly
simplified using the Faraday constant. This problem has been fixed
and the registry has been improved to prevent overwriting previously
registered names or symbols. F is now reserved for the farad.

Addition and subtraction of quantities with different but compatible
units is now supported. Additionally, changes have been made in the
forthcoming numpy-1.4.0 to extend this compatibility to other ufuncs
as well.

Numpydoc, an external package developed to extend Sphinx for the
numpy documentation project, is now required to build quantities'
documentation.

Bugs fixed
**********

* #495181 epsilon_0 simplifies incorrectly
* #490323 bad spelling for polarizabilities
