from __future__ import absolute_import

import numpy
from ..quantities import Quantity, dimensionless, radian, degree
from ..utilities import with_doc


#__all__ = [
#    'ceil', 'exp', 'expm1', 'floor', 'log', 'log10', 'log1p', 'log2', 'rint'
#]


@with_doc(numpy.prod)
def prod(a, axis=None, dtype=None, out=None):
    return a.prod(axis, dtype, out)

@with_doc(numpy.sum)
def sum(a, axis=None, dtype=None, out=None):
    return a.sum(axis, dtype, out)

@with_doc(numpy.nansum)
def nansum(a, axis=None):
    if not isinstance(a, Quantity):
        return numpy.nansum(a, axis)

    return Quantity(
        numpy.nansum(a.magnitude, axis),
        a.dimensionality,
        copy=False
    )

@with_doc(numpy.cumprod)
def cumprod(a, axis=None, dtype=None, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    return a.cumprod(axis, dtype, out)

@with_doc(numpy.cumsum)
def cumsum(a,axis=None, dtype=None, out=None):
    return a.cumsum(axis, dtype, out)

diff = numpy.diff

@with_doc(numpy.ediff1d)
def ediff1d(ary, to_end=None, to_begin=None):
    if not isinstance(ary, Quantity):
        return numpy.ediff1d(ary, to_end, to_begin)

    return Quantity(
        numpy.ediff1d(ary.magnitude, to_end, to_begin),
        ary.dimensionality,
        copy=False
    )

@with_doc(numpy.gradient)
def gradient(f, *varargs):
    # if no sample distances are specified, use dimensionless 1
    # this mimicks the behavior of numpy.gradient, but perhaps we should
    # remove this default behavior
    # removed for now::
    #
    #   if len(varargs) == 0:
    #       varargs = (Quantity(1),)

    varargsQuantities = [Quantity(i, copy=False) for i in varargs]
    varargsMag = tuple([i.magnitude for i in varargsQuantities])
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

@with_doc(numpy.cross)
def cross (a, b , axisa=-1, axisb=-1, axisc=-1, axis=None):
    if not (isinstance(a, Quantity) and isinstance(b, Quantity)):
        return numpy.cross(a, b, axisa, axisb, axisc, axis)

    if not isinstance(a, Quantity):
        a = Quantity(a, dimensionless, copy=False)
    if not isinstance(b, Quantity):
        b = Quantity(b, dimensionless, copy=False)

    return Quantity(
        numpy.cross(a, b, axisa, axisb, axisc, axis),
        a._dimensionality*b._dimensionality,
        copy=False
    )

@with_doc(numpy.trapz)
def trapz(y, x=None, dx=1.0, axis=-1):
    # this function has a weird input structure, so it is tricky to wrap it
    # perhaps there is a simpler way to do this
    if (
        not isinstance(y, Quantity)
        and not isinstance(x, Quantity)
        and not isinstance(dx, Quantity)
    ):
        return numpy.trapz(y, x, dx, axis)

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

@with_doc(numpy.exp)
def exp(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.exp(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.exp(x.magnitude), copy = False)

@with_doc(numpy.expm1)
def expm1(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.expm1(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.expm1(x.magnitude), copy = False)

@with_doc(numpy.log)
def log(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.log(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log(x.magnitude), copy = False)

@with_doc(numpy.log10)
def log10(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.log10(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log10(x.magnitude), copy = False)

@with_doc(numpy.log2)
def log2(x, y=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.log2(x, y)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log2(x.magnitude), copy = False)

@with_doc(numpy.log1p)
def log1p(x, out=None):
    if not isinstance(x, Quantity):
        return numpy.log1p(x, out)
    x = x.rescale(dimensionless)

    return Quantity(numpy.log1p(x.magnitude), copy = False)

rint = numpy.rint
floor = numpy.floor
ceil = numpy. ceil
prod = numpy.prod

@with_doc(numpy.sin)
def sin(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to radians.

    Returns a dimensionless quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.sin(x, out)

    return Quantity(numpy.sin(x.rescale(radian).magnitude, out),
                    copy=False)

@with_doc(numpy.arcsin)
def arcsin(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.

    Returns a quantity in units of radians.
    """
    if not isinstance(x, Quantity):
        return numpy.arcsin(x, out)

    return Quantity(
        numpy.arcsin(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

@with_doc(numpy.cos)
def cos(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to radians.

    Returns a dimensionless quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.cos(x, out)

    return Quantity(numpy.cos(x.rescale(radian).magnitude), copy=False)

@with_doc(numpy.arccos)
def arccos(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.

    Returns a quantity in units of radians.
    """
    if not isinstance(x, Quantity):
        return numpy.arccos(x, out)

    return Quantity(
        numpy.arccos(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

@with_doc(numpy.tan)
def tan(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to radians.

    Returns a dimensionless quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.tan(x, out)

    return Quantity(numpy.tan(x.rescale(radian).magnitude), copy=False)

@with_doc(numpy.arctan)
def arctan(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.

    Returns a quantity in units of radians.
    """
    if not isinstance(x, Quantity):
        return numpy.arctan(x, out)

    return Quantity(
        numpy.arctan(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

@with_doc(numpy.arctan2)
def arctan2(x1, x2, out=None):
    """
    Raises a ValueError if inputs do not have identical units.

    Returns a quantity in units of radians.
    """
    if not (isinstance(x1, Quantity) and isinstance(x2, Quantity)):
        return numpy.arctan2(x1, x2, out)

    if not isinstance(x1, Quantity):
        x1 = Quantity(x1, dimensionless, copy=False)
    if not isinstance(x2, Quantity):
        x2 = Quantity(x2, dimensionless, copy=False)

    if x1._dimensionality.simplified != x2._dimensionality.simplified:
        raise ValueError(
            'x1 and x2 must have identical units, got "%s" and "%s"'\
            % (str(x1._dimensionality), str(x2._dimensionality))
        )

    return Quantity(
        numpy.arctan2(x1.magnitude, x2.magnitude, out),
        radian,
        copy=False
    )

@with_doc(numpy.hypot)
def hypot(x1, x2, out = None):
    """
    Raises a ValueError if inputs do not have identical units.
    """
    if not (isinstance(x1, Quantity) and isinstance(x2, Quantity)):
        return numpy.hypot(x1, x2, out)

    if not isinstance(x1, Quantity):
        x1 = Quantity(x1, dimensionless, copy=False)
    if not isinstance(x2, Quantity):
        x2 = Quantity(x2, dimensionless, copy=False)

    if x1._dimensionality != x2._dimensionality:
        raise ValueError(
            'x1 and x2 must have identical units, got "%s" and "%s"'\
            % (str(x1._dimensionality), str(x2._dimensionality))
        )

    return Quantity(
        numpy.hypot(x1.magnitude, x2.magnitude, out),
        x1.dimensionality,
        copy = False
    )

@with_doc(numpy.degrees)
def to_degrees(x, out=None):
    """
    Requires input in units of radians to be compatible with
    numpy.degrees.

    For more flexible conversions of angular quantities,
    x.rescale(degrees) should be used instead.
    """
    if not isinstance(x, Quantity):
        return numpy.degrees(x, out)

    if x._dimensionality != radian._dimensionality:
        raise ValueError(
            'x must be in units of radians, got "%s"' % (str(x._dimensionality))
        )

    return Quantity(numpy.degrees(x.magnitude, out), degree, copy=False)

@with_doc(numpy.radians)
def to_radians(x, out=None):
    """
    Requires input in units of degrees to be compatible with
    numpy.radians.

    For more flexible conversions of angular quantities,
    x.rescale(radians) should be used instead.
    """
    if not isinstance(x, Quantity):
        return numpy.radians(x, out)

    if x._dimensionality != degree._dimensionality:
        raise ValueError(
            'x must be in units of degrees, got "%s"' % (str(x._dimensionality))
        )

    return Quantity(numpy.radians(x.magnitude, out), radian, copy=False)

@with_doc(numpy.unwrap)
def unwrap(p, discont=numpy.pi, axis=-1):
    if not (isinstance(p, Quantity) and isinstance(discont, Quantity)):
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

@with_doc(numpy.sinh)
def sinh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.sinh(x, out)

    return Quantity(
        numpy.sinh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(numpy.cosh)
def cosh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.cosh(x, out)

    return Quantity(
        numpy.cosh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(numpy.tanh)
def tanh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.tanh(x, out)

    return Quantity(
        numpy.tanh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(numpy.arcsinh)
def arcsinh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.arcsinh(x, out)

    return Quantity(
        numpy.arcsinh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(numpy.arccosh)
def arccosh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.arccosh(x, out)

    return Quantity(
        numpy.arccosh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(numpy.arctanh)
def arctanh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return numpy.arctanh(x, out)

    return Quantity(
        numpy.arctanh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )
