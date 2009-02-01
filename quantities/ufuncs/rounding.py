from __future__ import absolute_import

import numpy
from ..quantities import Quantity, dimensionless, radian, degree
from .utilities import usedoc

round = numpy.round

around = numpy.around
round_ = numpy.around
rint = numpy.rint

@usedoc(numpy.fix)
def fix(x , y = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.fix(x, out)
    
    return Quantity(numpy.fix(x.magnitude), x.dimensionality, copy = False)
 
floor = numpy.floor
ceil = numpy. ceil 