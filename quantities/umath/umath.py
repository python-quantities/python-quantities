from __future__ import absolute_import

import numpy
from ..quantities import Quantity, dimensionless, radian, degree
from ..utilities import usedoc


#__all__ = [
#    'ceil', 'exp', 'expm1', 'floor', 'log', 'log10', 'log1p', 'log2', 'rint'
#]


_check_dimensionless = \
"""    checks to make sure exponents are dimensionless so the operation
    makes sense"""



@usedoc(numpy.prod)
def prod (a, axis=None, dtype=None, out=None):
    #  Return the product of array elements over a given axis.
    return a.prod(axis, dtype, out)

@usedoc(numpy.sum)
def sum (a , axis=None, dtype=None, out=None):
    #  Return the sum of array elements over a given axis.
    return a.sum(axis, dtype, out)

@usedoc(numpy.nansum)
def nansum (a , axis=None):
    # Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
    if not isinstance(a, Quantity):
        return numpy.nansum(a, axis)

    return Quantity(numpy.nansum(a.magnitude, axis),
                    a.dimensionality,
                    copy=False)

@usedoc(numpy.cumprod, suffix = "\n a must be dimensionless")
def cumprod (a, axis=None, dtype=None, out=None) :
    #   Return the cumulative product of elements along a given axis.
    return a.cumprod(axis, dtype, out)

@usedoc(numpy.cumsum)
def cumsum (a ,axis=None, dtype=None, out=None):
    #  Return the cumulative sum of the elements along a given axis.
    return a.cumsum(axis, dtype, out)


diff = numpy.diff

@usedoc(numpy.ediff1d)
def ediff1d (ary , to_end=None, to_begin=None):
    #     The differences between consecutive elements of an array.
    if not isinstance(ary, Quantity):
        return numpy.ediff1d(ary, to_end, to_begin)

    return Quantity(numpy.ediff1d(ary.magnitude, to_end, to_begin),
                    ary.dimensionality,
                    copy=False)

@usedoc(numpy.gradient)
def gradient (f, *varargs):  #   Return the gradient of an N-dimensional array.

    # if no sample distances are specified, use dimensionless 1
    # this mimicks the behavior of numpy.gradient, but perhaps we should
    #remove this default behavior
    """# removed for now
    if len(varargs) == 0:
        varargs = (Quantity(1),)
    """
    #turn all the vararg elements into Quantities if they are not already
    varargsQuantities = [Quantity(i, copy = False) for i in varargs]

    # get the magnitudes for all the
    varargsMag = tuple([ i.magnitude for i in varargsQuantities])

    ret = numpy.gradient(f.magnitude, *varargsMag)

    if len(varargs) == 1:
        # if there was only one sample distance provided,
        # apply the units in all directions
        return tuple([ Quantity(i, f.units/varargs[0].units)  for i  in ret])
    else:
        #give each output array the units of the input array
        #divided by the units of the spacing quantity given
        return tuple([ Quantity(i, f.units/j.units)
                      for i,j  in zip( ret, varargsQuantities)])



@usedoc(numpy.cross)
def cross (a, b , axisa=-1, axisb=-1, axisc=-1, axis=None):
    #Return the cross product of two (arrays of) vectors.

    return Quantity(numpy.cross(a, b , axisa, axisb, axisc, axis),
                    a.units * b.units, copy = False)

@usedoc(numpy.trapz)
def trapz (y ,x=None, dx=1.0, axis=-1):
    #Integrate along the given axis using the composite trapezoidal rule.

    # this function has a weird input structure, so it is tricky to wrap it
    # perhaps there is a simpler way to do this

    # make sure this can be used as the normal numpy function
    if (not isinstance(y, Quantity)
        and not isinstance(x, Quantity)
        and not isinstance(dx, Quantity)):

        return numpy.trapz(y, x, dx, axis)
    # convert x, y and dx to Quantities
    if not isinstance(y, Quantity):
        y = Quantity(y, copy = False)


    if not isinstance(x, Quantity) and not x is None:
        x = Quantity(x, copy = False)

    if not isinstance(dx, Quantity):
        dx = Quantity(dx, copy = False)

    if x is None:
        ret = numpy.trapz(y.magnitude , x, dx.magnitude, axis)
        return Quantity ( ret, y.units * dx.units)
    else:
        ret = numpy.trapz(y.magnitude , x.magnitude, dx.magnitude, axis)
        return Quantity ( ret, y.units * x.units)



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

    if x.dimensionality != radian.dimensionality:
        raise ValueError("x must have units of radians, but has units of" +
                         str(x.dimensionality))

    return Quantity(numpy.degrees(x.magnitude, out), degree, copy=False)

def to_radians (x, out = None):
    """
    generally, x.rescale(radians) should be used instead
    """
    if not isinstance(x, Quantity):
        return numpy.radians(x, out)

    if x.dimensionality != degree.dimensionality:
        raise ValueError("x must have units of degree, but has units of" +
                         str(x.dimensionality))

    return Quantity(numpy.radians(x.magnitude, out), radian, copy=False)

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
