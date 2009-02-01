import numpy
import quantities
from quantities import Quantity, dimensionless
from decorators import usedoc



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
