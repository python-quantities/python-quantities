************
Known Issues
************

Quantities arrays are designed to work like normal numpy arrays. However, a few
operations are not yet fully functioning.

.. note:: 
    In the following code examples, it's assumed that you've initiated the
    following imports:
    
    >>> import numpy as np
    >>> import quantities as pq


Temperature conversion
======================

Quantities is not designed to handle coordinate systems that require a point of
reference, like positions on a map or absolute temperature scales. Proper 
support of coordinate systems would be a fairly large undertaking and is 
outside the scope of this project. Furthermore, consider the following::

  >>> T_0 = 100 * pq.K
  >>> T_1 = 200 * pq.K
  >>> dT = T_1-T_0
  >>> dT.units = pq.degF

To properly support the above example, quantities would have to distinguish
absolute temperatures with temperature differences. It would have to know how
to combine these two different animals, etc. The quantities project has 
therefore elected to limit the scope to relative quantities.

As a consequence, quantities treats temperatures as a temperature difference.
This is a distinction without a difference when considering Kelvin and Rankine,
or transformations between the two scales, since both scales have zero offset.
Temperature scales in Celsius and Fahrenheit are different and would require a 
non-zero offset, which is not supported in Quantities unit transformation 
framework. 


Trigonometric and other math functions
======================================

numpy's trigonometric ufuncs, such as `np.sin`, accept quantities in
radians or dimensionless units, and raise an error for any other units, even
other angular units::

  >>> np.sin(90 * pq.deg)
  Traceback (most recent call last):
    ...
  ValueError: expected units of radians, got "deg"

Either rescale the argument to radians first, or use the corresponding
quantities function, which accepts any angular units::

  >>> np.sin((90 * pq.deg).rescale(pq.rad))
  array(1.) * dimensionless
  >>> pq.sin(90 * pq.deg)
  array(1.) * dimensionless

For an exhaustive list of these functions, see `pq.umath`.

Similarly, `np.exp`, `np.log` and related functions require a dimensionless
argument. Quantities whose units cancel out but are not plain
`dimensionless`, such as `ms/s`, are currently rejected by these functions
and by the trigonometric ufuncs. Use `.simplified` to reduce them first::

  >>> np.sin((1 * pq.ms / pq.s).simplified)
  array(0.001) * dimensionless

numpy ufuncs that quantities does not know about raise a `ValueError`
instead of silently ignoring the units.


Functions which ignore/drop units
=================================

There are additional numpy functions not in `pq.umath` that ignore and drop
units. Below is a list of known functions in this category

* `arange`
* `interp`
