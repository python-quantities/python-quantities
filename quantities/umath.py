import numpy as np

from .quantity import Quantity
from .units import dimensionless, radian, degree  # type: ignore[no-redef]
from .decorators import with_doc


__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "cos",
    "cosh",
    "cross",
    "cumprod",
    "cumsum",
    "diff",
    "ediff1d",
    "gradient",
    "hypot",
    "nansum",
    "np",
    "prod",
    "sin",
    "sinh",
    "sum",
    "tan",
    "tanh",
    "trapz",
    "unwrap",
]


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
    # this mimics the behavior of np.gradient, but perhaps we should
    # remove this default behavior
    # removed for now::
    #
    #   if len(varargs) == 0:
    #       varargs = (Quantity(1),)

    varargsQuantities = [Quantity(i, copy=False) for i in varargs]
    varargsMag = tuple(i.magnitude for i in varargsQuantities)
    ret = np.gradient(f.magnitude, *varargsMag)

    if len(varargs) == 1:
        # if there was only one sample distance provided,
        # apply the units in all directions
        return tuple( Quantity(i, f.units/varargs[0].units)  for i  in ret)
    else:
        #give each output array the units of the input array
        #divided by the units of the spacing quantity given
        return tuple( Quantity(i, f.units/j.units)
                      for i,j  in zip( ret, varargsQuantities))

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


def trapz(y, x=None, dx=1.0, axis=-1):
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapz : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    See Also
    --------
    sum, cumsum

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.

    Docstring is from the numpy 1.26 code base
    https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/function_base.py#L4857-L4984


    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    """
    # this function has a weird input structure, so it is tricky to wrap it
    # perhaps there is a simpler way to do this
    if (
        not isinstance(y, Quantity)
        and not isinstance(x, Quantity)
        and not isinstance(dx, Quantity)
    ):
        return _trapz(y, x, dx, axis)

    if not isinstance(y, Quantity):
        y = Quantity(y, copy = False)
    if not isinstance(x, Quantity) and not x is None:
        x = Quantity(x, copy = False)
    if not isinstance(dx, Quantity):
        dx = Quantity(dx, copy = False)

    if x is None:
        ret = _trapz(y.magnitude , x, dx.magnitude, axis)
        return Quantity ( ret, y.units * dx.units)
    else:
        ret = _trapz(y.magnitude , x.magnitude, dx.magnitude, axis)
        return Quantity ( ret, y.units * x.units)

def _trapz(y, x, dx, axis):
    """ported from numpy 1.26 since it will be deprecated and removed"""
    try:
        # if scipy is available, we use it
        from scipy.integrate import trapezoid  # type: ignore
    except ImportError:
        # otherwise we use the implementation ported from numpy 1.26
        from numpy.core.numeric import asanyarray
        from numpy.core.umath import add
        y = asanyarray(y)
        if x is None:
            d = dx
        else:
            x = asanyarray(x)
            if x.ndim == 1:
                d = diff(x)
                # reshape to correct shape
                shape = [1]*y.ndim
                shape[axis] = d.shape[0]
                d = d.reshape(shape)
            else:
                d = diff(x, axis=axis)
        nd = y.ndim
        slice1 = [slice(None)]*nd
        slice2 = [slice(None)]*nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        try:
            ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
        except ValueError:
            # Operations didn't work, cast to ndarray
            d = np.asarray(d)
            y = np.asarray(y)
            ret = add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
        return ret
    else:
        return trapezoid(y, x=x, dx=dx, axis=axis)

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
