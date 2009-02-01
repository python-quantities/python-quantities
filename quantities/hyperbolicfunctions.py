import numpy
import quantities
from quantities import Quantity, dimensionless


def sinh(x, out = None):
    """
    checks to make sure exponents are dimensionless so the operation makes
    sense
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.sinh(x, out)

    x = x.rescale(dimensionless)

    return Quantity(numpy.sinh(x.magnitude, out),
                    dimensionless,
                    copy = False)

def cosh(x, out = None):
    """
    checks to make sure exponents are dimensionless so the operation makes
    sense
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.cosh(x, out)

    x = x.rescale(dimensionless)

    return Quantity(numpy.cosh(x.magnitude, out),
                    dimensionless,
                    copy = False)


def tanh(x, out = None):
    """
    checks to make sure exponents are dimensionless so the operation makes
    sense
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.tanh(x, out)

    x = x.rescale(dimensionless)

    return Quantity(numpy.tanh(x.magnitude, out),
                    dimensionless,
                    copy = False)

def arcsinh(x, out = None):
    """
    checks to make sure exponents are dimensionless so the operation makes
    sense
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.arcsinh(x, out)

    x = x.rescale(dimensionless)

    return Quantity(numpy.arcsinh(x.magnitude, out),
                    dimensionless,
                    copy = False)

def arccosh(x, out = None):
    """
    checks to make sure exponents are dimensionless so the operation makes
    sense
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.arccosh(x, out)

    x = x.rescale(dimensionless)

    return Quantity(numpy.arccosh(x.magnitude, out),
                    dimensionless,
                    copy = False)

def arctanh(x, out = None):
    """
    checks to make sure exponents are dimensionless so the operation makes
    sense
    """
    # we want this to be useable for both quantities are other types
    if not isinstance(x, Quantity):
        return numpy.arctanh(x, out)

    x = x.rescale(dimensionless)

    return Quantity(numpy.arctanh(x.magnitude, out),
                    dimensionless,
                    copy = False)