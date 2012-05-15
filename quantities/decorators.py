# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import os
import re
import string
import sys
import warnings
from functools import partial, wraps


def memoize(f, cache={}):
    @wraps(f)
    def g(*args, **kwargs):
        key = (f, tuple(args), frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key].copy()
    return g


class with_doc:

    """
    This decorator combines the docstrings of the provided and decorated objects
    to produce the final docstring for the decorated object.
    """

    def __init__(self, method, use_header=True):
        self.method = method
        if use_header:
            self.header = \
    """

    Notes
    -----
    """
        else:
            self.header = ''

    def __call__(self, new_method):
        new_doc = new_method.__doc__
        original_doc = self.method.__doc__
        header = self.header

        if original_doc and new_doc:
            new_method.__doc__ = """
    %s
    %s
    %s
        """ % (original_doc, header, new_doc)

        elif original_doc:
            new_method.__doc__ = original_doc

        return new_method

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

    from .quantity import Quantity

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
                args[i] = args[i].simplified

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
                                    .dimensionality.simplified
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
