from __future__ import absolute_import

import numpy
from ..quantities import Quantity
from ..utilities import with_doc


__all__ = ['fix']


@with_doc(numpy.fix)
def fix(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.fix(x, out)

    return Quantity(numpy.fix(x.magnitude), x.dimensionality, copy=False)
