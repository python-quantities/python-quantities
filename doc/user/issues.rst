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


`umath` functions
=================

Many common math functions ignore the dimensions of quantities. For example,
trigonometric functions (e.g. `np.sin`) suffer this fate. For these functions,
quantities arrays are treated like normal arrays and the calculations proceed
as normal (except that a "not implemented" warning is raised). Note, however,
this behavior is not ideal since some functions should behave differently for
different units. For example, you would expect `np.sin` to give different
results for an angle of 1° versus an angle of 1 radian; instead, `np.sin`
extracts the magnitude of the input and assumes that it is already in radians.

To properly handle quantities, use the corresponding quantities functions
whenever possible. For example, `pq.sin` will properly handle the angle inputs
described above. For an exhaustive list, see the functions defined in
`pq.umath`.


Functions which ignore/drop units
=================================

There are additional numpy functions not in `pq.umath` that ignore and drop
units. Below is a list known functions in this category

* `vstack`
* `interp`

