.. _credits:

*******
Credits
*******


Quantities was written by Darren Dale, with the hope that additional
developers will contribute to --- and take credit for --- its
development.

Andrew Davison has maintained the project since around 2019, reviewing and
merging contributions and fixing compatibility issues as Python and NumPy have
evolved (floordiv, parsing of unit strings, in-place arithmetic under NumPy
2.x), migrating the build system to pyproject.toml, and expanding CI coverage.

Special thanks to those who have made valuable contributions
(roughly in order of first contribution by date)

Charles Doutriaux
  wrote the python wrapper for unit conversions with the UDUnits
  library (no longer used)

Enthought Inc.
  the original units registry was developed at Enthought for the
  Enthought Tool Suite.

John Salvatier
  added support for the Quantity iterator, contributed to support for
  simplified representations of units, comparison operators, and
  contributed unit tests.

Tony Yu
  contributed several bug fixes and contributed to the documentation.

Kenneth D. Mankoff
  added nan-aware functions (nansum, nanmean, nanmin, nanmax, etc.)
  and sqrt support.

Thomas Hisch
  added CI support, new units (kilosecond, megasecond, decimeter,
  SI multiples for Kelvin, microcoulomb, millicoulomb), and
  ``__format__`` support for Quantity.

Tom Close
  added concentration units, additional litre aliases, electrical
  units (Farads, Siemens, Ohms), and corrected volume aliases.

Björn Dahlgren
  added comparison ufuncs, HTML dimensionality representation,
  additional math ufuncs, dtype preservation fixes, and NumPy 2.0
  support.

Hugo van Kemenade
  migrated CI from Travis to GitHub Actions, dropped Python 2
  support, and modernised the test setup.

Peter Konradi
  added type stubs (mypy annotations) for the core classes.

Zach McKenzie
  updated deprecated NumPy names, ported ``trapz`` to umath,
  fixed NumPy 2.0 array wrapping, and prevented arbitrary code
  evaluation in unit strings.

Additional contributions — including targeted bug fixes, unit
additions, and documentation improvements — have been made by
ac1982, Daniel McCloy, Daniel A. Wagenaar, Espen Hagen, jadoro,
Jakob Kaiser, Johannes Pfabe, Josef Kemetmüller, Justin Keogh,
lumbric, Mathieu Scheltienne, Michael von Papen, Michał Górny,
mjn, Olivier Sechet, p2jones, Philipp Rautenberg, Richard Gerkin,
Rob Falck, Robert Schütz, Ryan May, Ryan Senkbeil, Sai Asish Y,
Siyan Zhao, Takashi, Takumasa Nakamura, Tim Gates, wilsaj, and Yves Delley.
