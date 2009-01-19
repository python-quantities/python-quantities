import numpy
import quantities
from quantities import Quantity, dimensionless


def exp(x):
    """
    checks to make sure exponents are dimensionless so the operation makes
    sense
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.exp(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.exp(x.magnitude), copy = False)


def sin(x, out = None):
    """
    checks to see if arguments are angles
    returns a dimensionless quantity
    """

    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.sin(x, out)

    return Quantity(numpy.sin(x.rescale(quantities.radian).magnitude, out),
                    copy=False)

def arcsin(x, out = None):
    """
    checks to see if arguments are dimensionless
    returns an array of radians
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.arcsin(x, out)

    return Quantity(numpy.arcsin(x.rescale(dimensionless).magnitude, out),
                                quantities.radians,
                                copy=False)

def cos (x, out = None):
    """
    checks to see if arguments are angles
    returns a dimensionless quantity
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.cos(x, out)

    return Quantity(numpy.cos(x.rescale(quantities.radian).magnitude),
                    copy=False)

def arccos(x, out = None):
    """
    checks to see if arguments are dimensionless
    returns an array of radians
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.arccos(x, out)

    return Quantity(numpy.arccos(x.rescale(quantities.dimensionless).magnitude
                                 , out),
                                quantities.radians,
                                copy=False)

def tan (x, out = None):
    """
    checks to see if arguments are angles
    returns a dimensionless quantity
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.tan(x, out)

    return Quantity(numpy.tan(x.rescale(quantities.radian).magnitude),
                    copy=False)

def arctan(x, out = None):
    """
    checks to see if arguments are dimensionless
    returns an array of radians
    """

    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.arctan(x, out)

    return Quantity(numpy.arctan(x.rescale(quantities.dimensionless).magnitude
                                , out),
                                quantities.radians,
                                copy=False)

def arctan2(x1, x2, out = None):
    """
    checks to see if arguments are dimensionless
    returns an array of radians
    """


    # we want this to be useable for both quantities are other types
    if (not isinstance(x1, Quantity)) or not isinstance(x2, Quantity) :
        return numpy.arctan2(x1, x2, out)

    x2 = x2.rescale(x1.dimensionality)

    return Quantity(numpy.arctan2(x1.magnitude, x2.magnitude, out),
                    quantities.radians,
                    copy=False)

def hypot(x1, x2, out = None):
    """
    x1 and x2 must have the same units
    """
    # we want this to be useable for both quantities are other types
    if (not isinstance(x1, Quantity)) or not isinstance(x2, Quantity) :
        return numpy.hypot(x1, x2, out)

    if x1.dimensionality != x2.dimensionality:
        raise ValueError("x1 (" + str(x1.dimensionality) +
                         ") and x2 (" + str(x2.dimensionality) +
                         ") must have the same units")
    return Quantity(numpy.hypot(x1.magnitude, x2.magnitude, out),
                    x1.dimensionality, copy = False)


# this is a less confusing name, but there should probably be discussion about this (see next function)
def toDegrees (x, out = None):
    """


    generally, x.rescale(degrees) should be used instead
    """

    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.degrees(x, out)

    if x.dimensionality != quantities.radians.dimensionality:
        raise ValueError("x must have units of radians, but has units of" +
                         str(x.dimensionality))

    return Quantity(numpy.degrees(x.magnitude, out), quantities.degree,
                           copy = False)

# this function was originally called "radians" however, this conflicts with the unit called radians
def toRadians (x, out = None):
    """
    generally, x.rescale(radians) should be used instead
    """

    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.radians(x, out)

    if x.dimensionality != quantities.degree.dimensionality:
        raise ValueError("x must have units of degree, but has units of" +
                         str(x.dimensionality))

    return Quantity(numpy.radians(x.magnitude, out), quantities.radians,
                           copy = False)

def unwrap(p, discont=3.1415926535897931, axis=-1):
    """


    """
    # we want this to be useable for both quantities are other types
    if not isinstance(p, Quantity) and not isinstance(discont, Quantity):
        return numpy.unwrap(p, discont, axis)

    if not isinstance(p, Quantity):
        p = Quantity(p, copy=False)

    if not isinstance(discont, Quantity):
        discont = Quantity(discont, copy=False)

    discont = discont.rescale(p.units)

    return Quantity(
                    numpy.unwrap(p.magnitude, discont.magnitude,
                             axis),
                    p.units)