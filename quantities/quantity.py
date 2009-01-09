"""
"""

import copy

import numpy

from quantities.dimensionality import BaseDimensionality, \
    MutableDimensionality, ImmutableDimensionality
from quantities.registry import unit_registry

def prepare_compatible_units(s, o):
    try:
        ss, os = s.simplified, o.simplified
        assert ss.units == os.units
        return ss, os
    except AssertionError:
        raise ValueError(
            'can not compare quantities with units of %s and %s'\
            %(s.units, o.units)
        )


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

    def __new__(cls, data, units='', dtype='d', mutable=True, copy = True):
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data, dtype=dtype)

        if copy == True:
            data = data.copy()

        if isinstance(data, Quantity) and units:
            if isinstance(units, BaseDimensionality):
                units = str(units)
            data = data.rescale(units)

        # should this be a "cooperative super" call instead?
        ret = numpy.ndarray.__new__(
            cls,
            data.shape,
            data.dtype,
            buffer=data
        )
        ret.flags.writeable = mutable
        return ret

    def __init__(self, data, units='', dtype='d', mutable=True, copy = True):
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
            m = self.magnitude
            m *= sq.magnitude / osq.magnitude
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

        conversion = 1.0
        for u, d in self.dimensionality.iteritems():
            conversion = conversion * u.reference_quantity**d
        return self.magnitude * unit_registry['dimensionless'] * conversion


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
        # if the other is not a quantity, try to cast it to a dimensionless
        #quantity
        try:
            if not isinstance(other, Quantity):
                other = Quantity(other)
            dims = self.dimensionality  + other.dimensionality
            magnitude = self.magnitude + other.rescale(self.units).magnitude

            return Quantity(magnitude, dims, magnitude.dtype)
        except ValueError:
            raise ValueError(
                'can not add quantities of with units of %s and %s'\
                %(str(self), str(other))
            )


    def __radd__(self, other):
        # if the other is not a quantity,
        #try to cast it to a dimensionless
        #quantity
        try:
            if not isinstance(other, Quantity):
                other = Quantity(other)
            dims =  other.dimensionality + self.dimensionality
            magnitude = other.magnitude + self.rescale(other.units).magnitude

            return Quantity(magnitude, dims, magnitude.dtype)
        except ValueError:
            raise ValueError(
                'can not add quantities of with units of %s and %s'\
                %(str(other), str(self))
            )

    def __sub__(self, other):
        try:
            # if the other is not a quantity,
            # try to cast it to a dimensionless
            #quantity
            if not isinstance(other, Quantity):
                other = Quantity(other)
            dims = self.dimensionality - other.dimensionality
            magnitude = self.magnitude - other.rescale(self.units).magnitude

            return Quantity(magnitude, dims, magnitude.dtype)
        except ValueError:
            raise ValueError(
                'can not subtract quantities of with units of %s and %s'\
                %(str(self), str(other))
            )

    def __rsub__(self, other):
        try:
            # if the other is not a quantity,
            #try to cast it to a dimensionless
            #quantity
            if not isinstance(other, Quantity):
                other = Quantity(other)
            print other

            #we need to reverse these, that's why this needs it's own function
            dims =  other.dimensionality - self.dimensionality
            print (other.dimensionality)
            magnitude = other.magnitude - self.rescale(other.units).magnitude

            return Quantity(magnitude, dims, magnitude.dtype)
        except ValueError:
            raise ValueError(
                'can not subtract quantities of with units of %s and %s'\
                %(str(other), str(self))
            )

    def __mul__(self, other):

        try:
            dims = self.dimensionality * other.dimensionality
            magnitude = self.magnitude * other.magnitude
        except AttributeError:
            magnitude = self.magnitude * other
            dims = copy.copy(self.dimensionality)
        return Quantity(magnitude, dims, magnitude.dtype)

    def __truediv__(self, other):

        try:
            dims = self.dimensionality / other.dimensionality
            magnitude = self.magnitude / other.magnitude
        except AttributeError:
            magnitude = self.magnitude / other
            dims = copy.copy(self.dimensionality)
        return Quantity(magnitude, dims, magnitude.dtype)

    __div__ = __truediv__

    def __rmul__(self, other):
        # TODO: This needs to be properly implemented
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return other * self**-1

    __rdiv__ = __rtruediv__

    def __pow__(self, other):
        if isinstance(other, Quantity):
            #if we are raising a quantity to a quantity, make sure
            #it's dimensionless
            simplified = other.simplified
            if (simplified.dimensionality !=
                unit_registry['dimensionless'].dimensionality):
                raise ValueError("exponent is not dimensionless")

            #make sure the quantity is simplified
            other = simplified.magnitude

        assert isinstance(other, (numpy.ndarray, int, float, long))

        dims = self.dimensionality**other
        magnitude = self.magnitude**other
        return Quantity(magnitude, dims, magnitude.dtype)

    def __rpow__(self, other):

        simplified = self.simplified
        #make sure that if we are going to raise something to a Quantity
        # that the quantity is dimensionless
        if (simplified.dimensionality !=
            unit_registry['dimensionless'].dimensionality):
            raise ValueError("exponent is not dimensionless")

        return other**simplified.magnitude


    def __repr__(self):
        return '%s*%s'%(numpy.ndarray.__str__(self), self.units)

    __str__ = __repr__

    def __getitem__(self, key):
        return Quantity(self.magnitude[key], self.units)

    def __setitem__(self, key, value):
        if not isinstance(value, Quantity):
            value = Quantity(value)

        self.magnitude[key] = value.rescale(self.units).magnitude

    def __iter__(self):
        return QuantityIterator(self)

    def __lt__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude < os.magnitude

    def __le__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude <= os.magnitude

    def __eq__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude == os.magnitude

    def __ne__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude != os.magnitude

    def __gt__(self, other):
        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude > os.magnitude

    def __ge__(self, other):

        ss, os = prepare_compatible_units(self, other)
        return ss.magnitude >= os.magnitude

    # other numpy functionality

    #I don't think this implementation is particularly efficient,
    #perhaps there is something better
    def tolist(self):
        #first get a dummy array from the ndarray method
        work_list = self.magnitude.tolist()
        #now go through and replace all numbers with the appropriate Quantity
        self._tolist(work_list)
        return work_list

    def _tolist(self, work_list):
        #iterate through all the items in the list
        for i in range(len(work_list)):
            #if it's a list then iterate through that list
            if isinstance(work_list[i], list):
                self._tolist(work_list[i])
            else:
                #if it's a number then replace it
                # with the appropriate quantity
                work_list[i] = Quantity(work_list[i], self.dimensionality)

    #need to implement other Array conversion methods:
    # item, itemset, tofile, dump, astype, byteswap


    def sum(self, axis=None, dtype=None, out=None):
        return Quantity(self.magnitude.sum(axis, dtype, out),
                         self.dimensionality, copy = False)

    def fill(self, scalar):
        # the behavior of fill needs to be discussed in the future
        # particularly the fact that this will raise an error if fill (0)
        #is called (when we self is not dimensionless)

        if not isinstance (scalar, Quantity):
            scalar = Quantity(scalar, copy = False)

        if scalar.dimensionality == self.dimensionality:
            self.magnitude.fill(scalar.magnitude)
        else:
            raise ValueError("scalar must have the same units as self")


    #reshape works as intended
    #transpose works as intended
    #reshape works as intented
    #flatten works as expected
    #ravel works as expected
    #squeeze works as expected

    #take functions as intended


    def put (self, indicies, values, mode = 'raise'):
        """
        performs the equivalent of ndarray.put () but enforces units
        values - must be an Quantity with the same units as self
        """

        if isinstance(values, Quantity):
            #this currently checks to see if the quantities are identical
            # and may be revised in the future, pending discussion
            if values.dimensionality == self.dimensionality:
                self.magnitude.put(indicies, values, mode)
            else:
                raise ValueError ("values must have the same units as self")
        else:
            raise TypeError("values must be a Quantity")

    #repeat performs as expected


    #choose does not function correctly, and it is not clear
    # how it would function, so for now it will not be implemented

    #sort works as intended

    #argsort
    def argsort (self, axis=-1, kind='quick', order=None):
        return self.magnitude.argsort(axis, kind, order)

    def searchsorted(self,values, side='left'):

        # if the input is not a Quantity, convert it
        if not isinstance (values, Quantity):
            values =Quantity(values, copy = False)

        if values.dimensionality != self.dimensionality:
            raise ValueError("values does not have the same units as self")

        return self.magnitude.searchsorted(values.magnitude, side)


    def nonzero(self):
        return self.magnitude.nonzero()

    #compress works as intended

    #diagonal works as intended

    #array calculations

    def max(self, axis=None, out=None):
        return Quantity(self.magnitude.max(), self.dimensionality,
                         copy = False)

    #argmax works as intended

    def min(self, axis=None, out=None):
        return Quantity(self.magnitude.min(), self.dimensionality,
                         copy = False)

    def argmin (self,axis=None, out=None):
        return self.magnitude.argmin()

    def ptp(self, axis=None, out=None):
        return Quantity(self.magnitude.ptp(), self.dimensionality,
                         copy= False)

    def clip (self, min = None, max = None, out = None):
        if min is None or  max is None:
            raise ValueError("at least one of min or max must be set")
        else:
            #set min and max to their appropriate values
            if min is None: min = Quantity(-numpy.Inf, self.dimensionality)
            if max is None: max = Quantity(numpy.Inf , self.dimensionality)

        if not isinstance(min, Quantity) or not isinstance (max, Quantity):
            raise TypeError("both min and max must be Quantities")

        clipped = self.magnitude.clip(
                            min.rescale(self.dimensionality).magnitude,
                            max.rescale(self.dimensionality).magnitude, out)
        return Quantity(clipped, self.dimensionality, copy = False)

    # conj, and conjugate will not currently be implemented
    # because it is not settled how we want to deal with
    # complex numbers

    def round (self, decimals=0, out=None):
        return Quantity(self.magnitude.round(decimals, out),
                        self.dimensionality, copy = False)

    def trace (self, offset = 0 , axis1 = 0, axis2=1, dtype=None, out=None):
        return Quantity(self.magnitude.trace(offset, axis1, axis2, dtype, out)
                        , self.dimensionality, copy = False)

    # cumsum works as intended

    def mean (self, axis=None, dtype=None, out=None):
        return Quantity(self.magnitude.mean( axis, dtype, out),
                         self.dimensionality, copy = False)

    def var (self, axis=None, dtype=None, out=None):
        #just return the variance of the magnitude
        # with the correct units (squared)
        return Quantity(self.magnitude.var( axis, dtype, out),
                         self.dimensionality**2, copy = False)

    def std (self, axis=None, dtype=None, out=None):
        #just return the std of the magnitude
        return Quantity(self.magnitude.std( axis, dtype, out),
                         self.dimensionality, copy = False)

    def prod(self, axis=None, dtype=None, out=None):
        # this is a little tricky because we have to have the correct units
        # which depends on how many were multiplied together
        prod_length = 0
        if axis == None:
            #if an axis was not given, the whole array is being multiplied
            prod_length = self.size
        else:
            #else then it's jsut the size of the axis
            prod_length = self.shape[axis]

        return Quantity(self.magnitude.prod( axis, dtype, out),
        #needs to have the right units so raise to prod_length
                         self.dimensionality**prod_length, copy = False)


    def cumprod(self, axis=None, dtype=None, out=None):
        #cum prod can only be calculated when the quantity is dimensionless
        # otherwise,
        # result will have different dimensionality based on the position
        if (self.dimensionality ==
            unit_registry['dimensionless'].dimensionality):
            return Quantity(self.magnitude.cumprod(axis, dtype, out),
                            copy = False)
        else:
            raise ValueError("Quantity must be dimensionless,\
            try using .simplified property")

    # for now all() and any() will be left unimplemented because this is ambiguous

    #list of unsupported functions: [all, any conj, conjugate, choose]

    #TODO:
    #check/implement resize, swapaxis




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
                args[i] = args[i].magnitude

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
                kwargs[i] = kwargs[i].magnitude


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

