from __future__ import absolute_import

import numpy
from ..quantities import Quantity, dimensionless, radian
from ..utilities import usedoc


#__all__ = [
#    'ceil', 'exp', 'expm1', 'floor', 'log', 'log10', 'log1p', 'log2', 'rint'
#]


_check_dimensionless = \
"""    checks to make sure exponents are dimensionless so the operation
    makes sense"""

@usedoc(numpy.exp, suffix = _check_dimensionless)
def exp(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.exp(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.exp(x.magnitude), copy = False)


@usedoc(numpy.expm1, suffix = _check_dimensionless)
def expm1(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.expm1(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.expm1(x.magnitude), copy = False)


@usedoc(numpy.log, suffix = _check_dimensionless)
def log(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.log(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log(x.magnitude), copy = False)

@usedoc(numpy.log10, suffix = _check_dimensionless)
def log10(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.log10(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log10(x.magnitude), copy = False)

@usedoc(numpy.log2, suffix = _check_dimensionless)
def log2(x, y=None):
    if not isinstance(x, Quantity):
        return numpy.log2(x, y)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log2(x.magnitude), copy = False)

@usedoc(numpy.log1p, suffix = _check_dimensionless)
def log1p(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.log1p(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log1p(x.magnitude), copy = False)


rint = numpy.rint
floor = numpy.floor
ceil = numpy. ceil
prod = numpy.prod

@usedoc(numpy.sin, suffix="""checks to see if arguments are angles
    returns a dimensionless quantity""")
def sin(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.sin(x, out)

    return Quantity(numpy.sin(x.rescale(radian).magnitude, out),
                    copy=False)

@usedoc(numpy.arcsin, suffix="""checks to see if arguments are dimensionless
    returns an array of radians""")
def arcsin(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.arcsin(x, out)

    return Quantity(
        numpy.arcsin(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

def cos(x, out=None):
    """
    checks to see if arguments are angles
    returns a dimensionless quantity
    """
    if not isinstance(x, Quantity):
        return numpy.cos(x, out)

    return Quantity(numpy.cos(x.rescale(radian).magnitude), copy=False)

def arccos(x, out=None):
    """
    checks to see if arguments are dimensionless
    returns an array of radians
    """
    if not isinstance(x, Quantity):
        return numpy.arccos(x, out)

    return Quantity(
        numpy.arccos(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

def tan(x, out=None):
    """
    checks to see if arguments are angles
    returns a dimensionless quantity
    """
    if not isinstance(x, Quantity):
        return numpy.tan(x, out)

    return Quantity(numpy.tan(x.rescale(radian).magnitude), copy=False)

def arctan(x, out=None):
    """
    checks to see if arguments are dimensionless
    returns an array of radians
    """
    if not isinstance(x, Quantity):
        return numpy.arctan(x, out)

    return Quantity(
        numpy.arctan(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

def arctan2(x1, x2, out=None):
    """
    checks to see if arguments are dimensionless
    returns an array of radians
    """
    if (not isinstance(x1, Quantity)) or not isinstance(x2, Quantity) :
        return numpy.arctan2(x1, x2, out)

    x2 = x2.rescale(x1.dimensionality)

    return Quantity(
        numpy.arctan2(x1.magnitude, x2.magnitude, out),
        radian,
        copy=False
    )

def hypot(x1, x2, out = None):
    """
    x1 and x2 must have the same units
    """
    if (not isinstance(x1, Quantity)) or not isinstance(x2, Quantity) :
        return numpy.hypot(x1, x2, out)

    if x1.dimensionality != x2.dimensionality:
        raise ValueError("x1 (" + str(x1.dimensionality) +
                         ") and x2 (" + str(x2.dimensionality) +
                         ") must have the same units")
    return Quantity(
        numpy.hypot(x1.magnitude, x2.magnitude, out),
        x1.dimensionality,
        copy = False
    )

def to_degrees (x, out=None):
    """
    generally, x.rescale(degrees) should be used instead
    """
    if not isinstance(x, Quantity):
        return numpy.degrees(x, out)

    if x.dimensionality != quantities.radians.dimensionality:
        raise ValueError("x must have units of radians, but has units of" +
                         str(x.dimensionality))

    return Quantity(numpy.degrees(x.magnitude, out), degree, copy=False)

def to_radians (x, out = None):
    """
    generally, x.rescale(radians) should be used instead
    """
    if not isinstance(x, Quantity):
        return numpy.radians(x, out)

    if x.dimensionality != quantities.degree.dimensionality:
        raise ValueError("x must have units of degree, but has units of" +
                         str(x.dimensionality))

    return Quantity(numpy.radians(x.magnitude, out), radians, copy=False)

def unwrap(p, discont=3.1415926535897931, axis=-1):
    if not isinstance(p, Quantity) and not isinstance(discont, Quantity):
        return numpy.unwrap(p, discont, axis)

    if not isinstance(p, Quantity):
        p = Quantity(p, copy=False)

    if not isinstance(discont, Quantity):
        discont = Quantity(discont, copy=False)

    discont = discont.rescale(p.units)

    return Quantity(
        numpy.unwrap(p.magnitude, discont.magnitude, axis),
        p.units
    )
