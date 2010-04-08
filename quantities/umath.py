from __future__ import absolute_import

import numpy as np

from .quantity import Quantity
from .units import dimensionless, radian, degree
from .decorators import with_doc


#__all__ = [
#    'exp', 'expm1', 'log', 'log10', 'log1p', 'log2'
#]


@with_doc(np.prod)
def prod(a, axis=None, dtype=None, out=None):
    return a.prod(axis, dtype, out)

@with_doc(np.sum)
def sum(a, axis=None, dtype=None, out=None):
    return a.sum(axis, dtype, out)

@with_doc(np.nansum)
def nansum(a, axis=None):
    if not isinstance(a, Quantity):
        return np.nansum(a, axis)

    return Quantity(
        np.nansum(a.magnitude, axis),
        a.dimensionality,
        copy=False
    )

@with_doc(np.cumprod)
def cumprod(a, axis=None, dtype=None, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    return a.cumprod(axis, dtype, out)

@with_doc(np.cumsum)
def cumsum(a,axis=None, dtype=None, out=None):
    return a.cumsum(axis, dtype, out)

diff = np.diff

@with_doc(np.ediff1d)
def ediff1d(ary, to_end=None, to_begin=None):
    if not isinstance(ary, Quantity):
        return np.ediff1d(ary, to_end, to_begin)

    return Quantity(
        np.ediff1d(ary.magnitude, to_end, to_begin),
        ary.dimensionality,
        copy=False
    )

@with_doc(np.gradient)
def gradient(f, *varargs):
    # if no sample distances are specified, use dimensionless 1
    # this mimicks the behavior of np.gradient, but perhaps we should
    # remove this default behavior
    # removed for now::
    #
    #   if len(varargs) == 0:
    #       varargs = (Quantity(1),)

    varargsQuantities = [Quantity(i, copy=False) for i in varargs]
    varargsMag = tuple([i.magnitude for i in varargsQuantities])
    ret = np.gradient(f.magnitude, *varargsMag)

    if len(varargs) == 1:
        # if there was only one sample distance provided,
        # apply the units in all directions
        return tuple([ Quantity(i, f.units/varargs[0].units)  for i  in ret])
    else:
        #give each output array the units of the input array
        #divided by the units of the spacing quantity given
        return tuple([ Quantity(i, f.units/j.units)
                      for i,j  in zip( ret, varargsQuantities)])

@with_doc(np.cross)
def cross (a, b , axisa=-1, axisb=-1, axisc=-1, axis=None):
    if not (isinstance(a, Quantity) and isinstance(b, Quantity)):
        return np.cross(a, b, axisa, axisb, axisc, axis)

    if not isinstance(a, Quantity):
        a = Quantity(a, dimensionless, copy=False)
    if not isinstance(b, Quantity):
        b = Quantity(b, dimensionless, copy=False)

    return Quantity(
        np.cross(a, b, axisa, axisb, axisc, axis),
        a._dimensionality*b._dimensionality,
        copy=False
    )

@with_doc(np.trapz)
def trapz(y, x=None, dx=1.0, axis=-1):
    # this function has a weird input structure, so it is tricky to wrap it
    # perhaps there is a simpler way to do this
    if (
        not isinstance(y, Quantity)
        and not isinstance(x, Quantity)
        and not isinstance(dx, Quantity)
    ):
        return np.trapz(y, x, dx, axis)

    if not isinstance(y, Quantity):
        y = Quantity(y, copy = False)
    if not isinstance(x, Quantity) and not x is None:
        x = Quantity(x, copy = False)
    if not isinstance(dx, Quantity):
        dx = Quantity(dx, copy = False)

    if x is None:
        ret = np.trapz(y.magnitude , x, dx.magnitude, axis)
        return Quantity ( ret, y.units * dx.units)
    else:
        ret = np.trapz(y.magnitude , x.magnitude, dx.magnitude, axis)
        return Quantity ( ret, y.units * x.units)

@with_doc(np.sin)
def sin(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to radians.

    Returns a dimensionless quantity.
    """
    if not isinstance(x, Quantity):
        return np.sin(x, out)

    return Quantity(np.sin(x.rescale(radian).magnitude, out),
                    copy=False)

@with_doc(np.arcsin)
def arcsin(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.

    Returns a quantity in units of radians.
    """
    if not isinstance(x, Quantity):
        return np.arcsin(x, out)

    return Quantity(
        np.arcsin(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

@with_doc(np.cos)
def cos(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to radians.

    Returns a dimensionless quantity.
    """
    if not isinstance(x, Quantity):
        return np.cos(x, out)

    return Quantity(np.cos(x.rescale(radian).magnitude), copy=False)

@with_doc(np.arccos)
def arccos(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.

    Returns a quantity in units of radians.
    """
    if not isinstance(x, Quantity):
        return np.arccos(x, out)

    return Quantity(
        np.arccos(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

@with_doc(np.tan)
def tan(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to radians.

    Returns a dimensionless quantity.
    """
    if not isinstance(x, Quantity):
        return np.tan(x, out)

    return Quantity(np.tan(x.rescale(radian).magnitude), copy=False)

@with_doc(np.arctan)
def arctan(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.

    Returns a quantity in units of radians.
    """
    if not isinstance(x, Quantity):
        return np.arctan(x, out)

    return Quantity(
        np.arctan(x.rescale(dimensionless).magnitude, out),
        radian,
        copy=False
    )

@with_doc(np.arctan2)
def arctan2(x1, x2, out=None):
    """
    Raises a ValueError if inputs do not have identical units.

    Returns a quantity in units of radians.
    """
    if not (isinstance(x1, Quantity) and isinstance(x2, Quantity)):
        return np.arctan2(x1, x2, out)

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
        np.arctan2(x1.magnitude, x2.magnitude, out),
        radian,
        copy=False
    )

@with_doc(np.hypot)
def hypot(x1, x2, out = None):
    """
    Raises a ValueError if inputs do not have identical units.
    """
    if not (isinstance(x1, Quantity) and isinstance(x2, Quantity)):
        return np.hypot(x1, x2, out)

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
        np.hypot(x1.magnitude, x2.magnitude, out),
        x1.dimensionality,
        copy = False
    )

@with_doc(np.unwrap)
def unwrap(p, discont=np.pi, axis=-1):
    if not (isinstance(p, Quantity) and isinstance(discont, Quantity)):
        return np.unwrap(p, discont, axis)

    if not isinstance(p, Quantity):
        p = Quantity(p, copy=False)
    if not isinstance(discont, Quantity):
        discont = Quantity(discont, copy=False)

    discont = discont.rescale(p.units)

    return Quantity(
        np.unwrap(p.magnitude, discont.magnitude, axis),
        p.units
    )

@with_doc(np.sinh)
def sinh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return np.sinh(x, out)

    return Quantity(
        np.sinh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(np.cosh)
def cosh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return np.cosh(x, out)

    return Quantity(
        np.cosh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(np.tanh)
def tanh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return np.tanh(x, out)

    return Quantity(
        np.tanh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(np.arcsinh)
def arcsinh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return np.arcsinh(x, out)

    return Quantity(
        np.arcsinh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(np.arccosh)
def arccosh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return np.arccosh(x, out)

    return Quantity(
        np.arccosh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )

@with_doc(np.arctanh)
def arctanh(x, out=None):
    """
    Raises a ValueError if input cannot be rescaled to a dimensionless
    quantity.
    """
    if not isinstance(x, Quantity):
        return np.arctanh(x, out)

    return Quantity(
        np.arctanh(x.rescale(dimensionless).magnitude, out),
        dimensionless,
        copy=False
    )
