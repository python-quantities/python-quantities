from __future__ import absolute_import

import numpy
from ..quantities import Quantity, dimensionless
from ..utilities import usedoc


__all__ = [
    'ceil', 'exp', 'expm1', 'log', 'log10', 'log1p', 'log2', 'rint', 'floor'
]


_check_dimensionless = \
"""    checks to make sure exponents are dimensionless so the operation
    makes sense"""

@usedoc(numpy.exp, suffix = _check_dimensionless)
def exp(x, out = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.exp(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.exp(x.magnitude), copy = False)


@usedoc(numpy.expm1, suffix = _check_dimensionless)
def expm1(x , out = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.expm1(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.expm1(x.magnitude), copy = False)


@usedoc(numpy.log, suffix = _check_dimensionless)
def log(x, out = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.log(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log(x.magnitude), copy = False)

@usedoc(numpy.log10, suffix = _check_dimensionless)
def log10(x, out = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.log10(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log10(x.magnitude), copy = False)

@usedoc(numpy.log2, suffix = _check_dimensionless)
def log2(x, y = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.log2(x, y)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log2(x.magnitude), copy = False)

@usedoc(numpy.log1p, suffix = _check_dimensionless)
def log1p(x, out = None):
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.log1p(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log1p(x.magnitude), copy = False)


rint = numpy.rint
floor = numpy.floor
ceil = numpy. ceil
