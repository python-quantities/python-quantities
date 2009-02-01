from __future__ import absolute_import

import numpy
from ..quantities import Quantity
from ..utilities import usedoc


__all__ = ['fix']


@usedoc(numpy.fix)
def fix(x , y = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.fix(x, out)

    return Quantity(numpy.fix(x.magnitude), x.dimensionality, copy = False)
