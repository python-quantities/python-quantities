"""
"""

import copy

import numpy

from quantities.dimensionality import BaseDimensionality, \
    MutableDimensionality, ImmutableDimensionality
from quantities.registry import unit_registry


class QuantityIterator:

    """an iterator for quantity objects"""

    def __init__(self, object):
        self.object = object
        self.iterator = super(Quantity, object).__iter__()

    def __iter__(self):
        return self

    def next(self):
        return Quantity(self.iterator.next(), self.object.units)


class Quantity(numpy.ndarray):

    # TODO: what is an appropriate value?
    __array_priority__ = 21

    def __new__(cls, data, units='', dtype='d', mutable=True):
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data, dtype=dtype)

        data = data.copy()

        if isinstance(data, Quantity) and units:
            if isinstance(units, BaseDimensionality):
                units = str(units)
            data = data.rescale(units)

        ret = numpy.ndarray.__new__(
            cls,
            data.shape,
            data.dtype,
            buffer=data
        )
        ret.flags.writeable = mutable
        return ret

    def __init__(self, data, units='', dtype='d', mutable=True):
        if isinstance(data, Quantity) and not units:
            dims = data.dimensionality
        elif isinstance(units, str):
            if units == '': units = 'dimensionless'
            dims = unit_registry[units].dimensionality
        elif isinstance(units, Quantity):
            dims = units.dimensionality
        elif isinstance(units, (BaseDimensionality, dict)):
            dims = units
        else:
            assert units is None
            dims = None

        self._mutable = mutable
        if self.is_mutable:
            if dims is None: dims = {}
            self._dimensionality = MutableDimensionality(dims)
        else:
            if dims is None:
                self._dimensionality = None
            else:
                self._dimensionality = ImmutableDimensionality(dims)

    @property
    def dimensionality(self):
        if self._dimensionality is None:
            return ImmutableDimensionality({self:1})
        else:
            return ImmutableDimensionality(self._dimensionality)

    @property
    def magnitude(self):
        return self.view(type=numpy.ndarray)

    @property
    def is_mutable(self):
        return self._mutable

    @property
    def udunits(self):
        return self.dimensionality.udunits

    # get and set methods for the units property
    def get_units(self):
        return str(self.dimensionality)
    def set_units(self, units):
        if not self.is_mutable:
            raise AttributeError("can not modify protected units")
        if isinstance(units, str):
            units = unit_registry[units]
        if isinstance(units, Quantity):
            try:
                assert units.magnitude == 1
            except AssertionError:
                raise ValueError('units must have unit magnitude')
        try:
            sq = Quantity(1.0, self.dimensionality).simplified
            osq = units.simplified
            assert osq.dimensionality == sq.dimensionality
            self.magnitude.flat[:] *= sq.magnitude.flat[:] / osq.magnitude.flat[:]
            self._dimensionality = \
                MutableDimensionality(units.dimensionality)
        except AssertionError:
            raise ValueError(
                'Unable to convert between units of "%s" and "%s"'
                %(sq.units, osq.units)
            )
    units = property(get_units, set_units)

    def rescale(self, units):
        """
        Return a copy of the quantity converted to the specified units
        """
        copy = Quantity(self)
        copy.units = units
        return copy

    @property
    def simplified(self):
        rq = self.magnitude * unit_registry['dimensionless']
        for u, d in self.dimensionality.iteritems():
            rq = rq * u.reference_quantity**d
        return rq

    def __array_finalize__(self, obj):
        self._dimensionality = getattr(
            obj, 'dimensionality', MutableDimensionality()
        )

#    def __deepcopy__(self, memo={}):
#        dimensionality = copy.deepcopy(self.dimensionality)
#        return self.__class__(
#            self.view(type=ndarray),
#            self.dtype,
#            dimensionality
#        )

#    def __cmp__(self, other):
#        raise

    def __add__(self, other):
        if self.dimensionality:
            assert isinstance(other, Quantity)
        dims = self.dimensionality  + other.dimensionality
        magnitude = self.magnitude + other.rescale(self.units).magnitude
        return Quantity(magnitude, dims, magnitude.dtype)

    def __sub__(self, other):
        if self.dimensionality:
            assert isinstance(other, Quantity)
        dims = self.dimensionality - other.dimensionality
        magnitude = self.magnitude - other.rescale(self.units).magnitude
        return Quantity(magnitude, dims, magnitude.dtype)

    def __mul__(self, other):
        assert isinstance(other, (numpy.ndarray, list , long, int, float))
        try:
            dims = self.dimensionality * other.dimensionality
            magnitude = self.magnitude * other.magnitude
        except:
            dims = copy.copy(self.dimensionality)
            magnitude = self.magnitude * other
        return Quantity(magnitude, dims, magnitude.dtype)

    def __truediv__(self, other):
        assert isinstance(other, (numpy.ndarray, list, long, int, float))
        try:
            dims = self.dimensionality / other.dimensionality
            magnitude = self.magnitude / other.magnitude
        except:
            dims = copy.copy(self.dimensionality)
            magnitude = self.magnitude / other
        return Quantity(magnitude, dims, magnitude.dtype)

    __div__ = __truediv__

    def __rmul__(self, other):
        # TODO: This needs to be properly implemented
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return other * self**-1

    __rdiv__ = __rtruediv__

    def __pow__(self, other):
        assert isinstance(other, (numpy.ndarray, int, float))
        dims = self.dimensionality**other
        magnitude = self.magnitude**other
        return Quantity(magnitude, dims, magnitude.dtype)

    def __repr__(self):
        return '%s*%s'%(numpy.ndarray.__str__(self), self.units)

    __str__ = __repr__

    def __getitem__(self, key):
        return Quantity(self.magnitude[key], self.units)

    def __setitem__(self, key, value):
        self.magnitude[key] = value.rescale(self.units).magnitude

    def __iter__(self):
        return QuantityIterator(self)

    def __lt__(self, other):
        try:
            ss, os = self.simplified, other.simplified
            assert ss.units == os.units
            return ss.magnitude < os.magnitude
        except AssertionError:
            raise ValueError(
                'can not compare quantities with units of %s and %s'\
                %(ss.units, os.units)
            )

    def __le__(self, other):
        try:
            ss, os = self.simplified, other.simplified
            assert ss.units == os.units
            return ss.magnitude <= os.magnitude
        except AssertionError:
            raise ValueError(
                'can not compare quantities with units of %s and %s'\
                %(ss.units, os.units)
            )

    def __eq__(self, other):
        try:
            ss, os = self.simplified, other.simplified
            assert ss.units == os.units
            return ss.magnitude == os.magnitude
        except AssertionError:
            raise ValueError(
                'can not compare quantities with units of %s and %s'\
                %(ss.units, os.units)
            )

    def __ne__(self, other):
        try:
            ss, os = self.simplified, other.simplified
            assert ss.units == os.units
            return ss.magnitude != os.magnitude
        except AssertionError:
            raise ValueError(
                'can not compare quantities with units of %s and %s'\
                %(ss.units, os.units)
            )

    def __gt__(self, other):
        try:
            ss, os = self.simplified, other.simplified
            assert ss.units == os.units
            return ss.magnitude > os.magnitude
        except AssertionError:
            raise ValueError(
                'can not compare quantities with units of %s and %s'\
                %(ss.units, os.units)
            )

    def __ge__(self, other):

        other = other.rescale(self.units)
        return self.magnitude >= other.magnitude
        try:
            ss, os = self.simplified, other.simplified
            assert ss.units == os.units
            return ss.magnitude >= os.magnitude
        except AssertionError:
            raise ValueError(
                'can not compare quantities with units of %s and %s'\
                %(ss.units, os.units)
            )



def quantitizer(base_function,
                handler_function = lambda *args, **kwargs: 1.0):
    """
    wraps a function so that it works properly with physical quantities
    (Quantities).
    arguments:
        base_function - the function to be wrapped
        handler_function - a function which takes the same arguments as the
            base_function  and returns a Quantity (or tuple of Quantities)
            which has (have) the units that the output of base_function should
            have.
        returns:
            a wrapped version of base_function that takes the same arguments
            and works with physical quantities. It will have almost the same
            __name__ and almost the same __doc__.
    """
    # define a function which will wrap the base function so that it works
    # with Quantities
    def wrapped_function(*args , **kwargs):

        # run the arguments through the handler function, this should
        # return a tuple of Quantities which have the correct units
        # for the output of the function we are wrapping
        handler_quantities= handler_function( *args, **kwargs)

        # now we need to turn Quantities into ndarrays so they behave
        # correctly
        #
        # first we simplify all units so that  addition and subtraction work
        # there may be another way to ensure this, but I do not have any good
        # ideas

        # in order to modify the args tuple, we have to turn it into a list
        args = list(args)

        #replace all the quantities in the argument list with ndarrays
        for i in range(len(args)):
            #test if the argument is a quantity
            if isinstance(args[i], Quantity):
                #convert the units to the base units
                args[i] = args[i].simplified()

                #view the array as an ndarray
                args[i] = args[i].view(type=numpy.ndarray)

        #convert the list back to a tuple so it can be used as an output
        args = tuple (args)

        #repalce all the quantities in the keyword argument
        #dictionary with ndarrays
        for i in kwargs:
            #test if the argument is a quantity
            if isinstance(kwargs[i], Quantity):
                #convert the units to the base units
                kwargs[i] = kwargs[i].simplifed()

                #view the array as an ndarray
                kwargs[i] = kwargs[i].view(type=numpy.ndarray)


        #get the result for the function
        result = base_function( *args, **kwargs)

        # since we have to modify the result, convert it to a list
        result = list(result)

        #iterate through the handler_quantities and get the correct
        # units


        length = min(   len(handler_quantities)   ,    len(result)   )

        for i in range(length):
            # if the output of the handler is a quantity make the
            # output of the wrapper function be a quantity with correct
            # units
            if isinstance(handler_quantities[i], Quantity):
                # the results should have simplified units since that's what
                # the inputs were (they were simplified earlier)
                # (reasons why this would not be true?)
                result[i] = Quantity(
                                result[i],
                                handler_quantities[i]
                                    .dimensionality.simplified()
                                    )
                #now convert the quantity to the appropriate units
                result[i] = result[i].rescale(
                                        handler_quantities[i].dimensionality)

        #need to convert the result back to a tuple
        result = tuple(result)
        return result

    # give the wrapped function a similar name to the base function
    wrapped_function.__name__ = base_function.__name__ + "_QWrap"
    # give the wrapped function a similar doc string to the base function's
    # doc string but add an annotation to the beginning
    wrapped_function.__doc__ = (
            "this function has been wrapped to work with Quantities\n"
            + base_function.__doc__)

    return wrapped_function
