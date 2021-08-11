"""
Quantities is designed to handle arithmetic and conversions of
physical quantities, which have a magnitude, dimensionality specified
by various units, and possibly an uncertainty. Quantities is designed
to work with numpy's standard ufuncs, many of which are already
supported. The package is actively developed, and while the current
features and API are stable, test coverage is incomplete and so the
package is not suggested for production use.

It is strongly suggested to import quantities to its own namespace, so
units and constants variables are not accidentally overwritten::

   >>> import quantities as pq

Here pq stands for "physical quantities" or "python quantities".
There are a number of ways to create a quantity. In practice, it is
convenient to think of quantities as a combination of a magnitude and
units. These two quantities are equivalent::

   >>> import numpy as np
   >>> q = np.array([1,2,3]) * pq.J
   >>> q = [1,2,3] * pq.J
   >>> print q
   [ 1.  2.  3.] J

The Quantity constructor can also be used to create quantities,
similar to numpy.array. Units can be designated using a string
containing standard unit abbreviations or unit names. For example::

   >>> q = pq.Quantity([1,2,3], 'J')
   >>> q = pq.Quantity([1,2,3], 'joules')

Units are also available as variables, and can be passed to
Quantity::

   >>> q = pq.Quantity([1,2,3], pq.J)

You can modify a quantity's units in place::

   >>> q = 1 * pq.m
   >>> q.units = pq.ft
   >>> print q
   3.280839895013123 ft

or equivalently::

   >>> q = 1 * pq.meter
   >>> q.units = 'ft' # or 'foot' or 'feet'
   >>> print q
   3.280839895013123 ft

Note that, with strings, units can be designated using plural
variants. Plural variants of the module variables are not available at
this time, in the interest of keeping the units namespace somewhat
manageable. `q.units = 'feet'` will work, `q.units = pq.feet` will
not.

The units themselves are special objects that can not be modified in
place::

   >>> pq.meter.units = 'feet'
   AttributeError: can not modify protected units

Instead of modifying a quantity in place, you can create a new
quantity, rescaled to the new units::

   >>> q = 300 * pq.ft * 600 * pq.ft
   >>> q2 = q.rescale('US_survey_acre')
   >>> print q2
   4.13221487605 US_survey_acre

but rescaling will fail if the requested units fails a dimensional
analysis::

   >>> q = 10 * pq.joule
   >>> q2 = q.rescale(pq.watt)
   ValueError: Unable to convert between units of "J" and "W"

Quantities can not be rescaled in place if the unit conversion fails
a dimensional analysis::

   >>> q = 10 * pq.joule
   >>> q.units = pq.watts
   ValueError: Unable to convert between units of "J" and "W"
   >>> print q
   10.0 J

Quantities will attempt to simplify units when the users intent is
unambiguous:

   >>> q = (10 * pq.meter)**3
   >>> q2 = q/(5*pq.meter)**2
   >>> print q2
   40 m

Quantities will not try to guess in an ambiguous situation:

   >>> q = (10 * pq.meter)**3
   >>> q2 = q/(5*pq.ft)**2
   >>> print q2
   40 m**3/ft**2

In that case, it is not clear whether the user wanted ft converted to
meters, or meters to feet, or neither. Instead, you can obtain a new
copy of the quantity in its irreducible units, which by default are SI
units::

   >>> q = (10 * pq.meter)**3
   >>> q2 = q/(5*pq.ft)**2
   >>> print q2
   40 m**3/ft**2
   >>> qs = q2.simplified
   >>> print qs
   430.556416668 m

It is also possible to customize the units in which simplified
quantities are expressed::

   >>> pq.set_default_units('cgs')
   >>> print pq.J.simplified
   10000000.0 g*cm**2/s**2
   >>> pq.set_default_units(length='m', mass='kg')

There are times when you may want to treat a group of units as a
single compound unit. For example, surface area per unit volume is a
fairly common quantity in materials science. If expressed in the
usual way, the quantity will be expressed in units that you may not
recognize::

   >>> q = 1 * pq.m**2 / pq.m**3
   >>> print q
   1.0 1/m

Here are some tricks for working with these compound units, which
can be preserved::

   >>> q = 1 * pq.CompoundUnit("m**2/m**3")
   >>> print q
   1.0 (m**2/m**3)

and can be simplified::

   >>> qs = q.simplified
   >>> qs
   1.0 1/m

and then rescaled back into compound units::

   >>> q2 = qs.rescale(CompoundUnit("m**2/m**3"))
   >>> print q2
   1.0 (m**2/m**3)

Compound units can be combined with regular units as well:

   >>> q = 1 * pq.CompoundUnit('parsec/cm**3') * pq.cm**2
   >>> print q
   1.0 cm**2*(parsec/cm**3)

It is easy to define a unit that is not already provided by
quantities. For example::

   >>> uK = pq.UnitQuantity('microkelvin', pq.degK/1e6, symbol='uK')
   >>> print uK
   1 uK (microkelvin)
   >>> q = 1000*uK
   >>> print q.simplified
   0.001 K

There is also support for quantities with uncertainty::

   >>> q = UncertainQuantity(4,J,.2)
   >>> q
   4.0*J
   +/-0.2*J (1 sigma)

By assuming that the uncertainties are uncorrelated, the uncertainty can be
propagated during arithmetic operations::

   >>> length = UncertainQuantity(2.0, m, .001)
   >>> width = UncertainQuantity(3.0, m, .001)
   >>> area = length*width
   >>> area
   6.0*m**2
   +/-0.00360555127546*m**2 (1 sigma)

In that case, the measurements of the length and width were independent, and
the two uncertainties presumed to be uncorrelated. Here is a warning though:

   >>> q*q
   16.0*J**2
   +/-1.1313708499*J**2 (1 sigma)

This result is probably incorrect, since it assumes the uncertainties of the two
multiplicands are uncorrelated. It would be more accurate in this case to use::

   >>> q**2
   16.0*J**2
   +/-1.6*J**2 (1 sigma)

There is an entire subpackage dedicated to physical constants. The
values of all the constants are taken from values published by the
National Institute of Standards and Technology at
http://physics.nist.gov/constants . Most physical constants have some
form of uncertainty, which has also been published by NIST. All
uncertainties are one standard deviation. There are lots of constants
and quantities includes them all (with one exception: F*, the Faraday
constant for conventional electrical current, which is defined in
units of C_90, for which I have not found a hard reference value).
Physical constants are sort of similar to compound units, for example:

   >>> print pq.constants.proton_mass
   1 m_p (proton_mass)
   >>> print pq.constants.proton_mass.simplified
   1.672621637e-27 kg
   +/-8.3e-35 kg (1 sigma)

A Latex representation of the dimensionality may be obtained in the following fashion::

    >>> g = pq.Quantity(9.80665,'m/s**2')
    >>> mass = 50 * pq.kg
    >>> weight = mass*g
    >>> print weight.dimensionality.latex
    $\\mathrm{\\frac{kg{\\cdot}m}{s^{2}}}$
    >>> weight.units = pq.N
    >>> print weight.dimensionality.latex
    $\\mathrm{N}$

The Latex output is compliant with the MathText subset used by Matplotlib.  To add
formatted units to the axis label of a Matplotlib figure, one could use::

    >>> ax.set_ylabel('Weight ' + weight.dimensionality.latex)

Greater customization is available via the markup.format_units_latex function.  It allows
the user to modify the font, the multiplication symbol, or to encapsulate the latex
string in parentheses.  Due to the complexity of CompoundUnits, the latex rendering
of CompoundUnits will utilize the latex \\frac{num}{den} construct.

Although it is not illustrated in this guide, unicode symbols can be
used to provide a more compact representation of the units. This
feature is disabled by default. It can be enabled by setting the
following in your ~/.pythonrc.py::

   quantities_unicode = True

or you can change this setting on the fly by doing::

   from quantities import markup
   markup.config.use_unicode = True # or False

Even when unicode is enabled, when you pass strings to designate
units, they should still conform to valid python expressions.

.. attention::
   Quantities is not a package for describing coordinate systems that require a
   point of reference, like positions on a map. In particular, Quantities does
   not support absolute temperature scales. Instead, temperatures are assumed to
   be temperature *differences*. For example:

   >>> T = 20 * pq.degC
   >>> print T.rescale('K')
   20.0 K

   Proper support of coordinate systems would be a fairly large undertaking and
   is outside the scope of this project.

"""

from __future__ import absolute_import

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .registry import unit_registry

from . import quantity
from .quantity import Quantity

from . import uncertainquantity
from .uncertainquantity import UncertainQuantity

from . import unitquantity
from .unitquantity import *

from .units import *

from . import constants

from .umath import *

def test(verbosity=1):
    import unittest
    suite = unittest.TestLoader().discover('quantities')
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
