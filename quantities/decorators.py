# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import os
import re
import string
import sys
import warnings
from functools import partial

from .quantity import Quantity
from .registry import unit_registry


def rescale_units(*q_args, **q_kwargs):
    """
    a decorator that will rescale the input to the specified units
    or raise an error. For example, a method with signature::

      @rescale_units(None, 'eV', 'm', c='K', d='dimensionless')
      def foo(self, a, b, c=1, d=1)

    will skip rescaling the input units for self, because "None" was
    specified. d will not be skipped, but be rescaled to be
    dimensionless. It is important to specify "None" for the self
    argument passed to methods (there is not a clean way to determine
    if the wrapped item is a function or a method). 

    """
    q_args = list(q_args)
    for i, q in enumerate(q_args):
        if isinstance(q, str):
            q_args[i] = unit_registry[q]

    for k, q in q_kwargs.iteritems():
        if isinstance(q, str):
            q_kwargs[k] = unit_registry[q]

    @decorator
    def wrapped(f, *args, **kwargs):
        try:
            args = list(args)
            for i, (v, q) in enumerate(zip(args, wrapped.args)):
                if q is None: continue
                if v.units != q.units:
                    args[i] = v.rescale(q.units)
            for k, v in kwargs.iteritems():
                q = wrapped.kwargs[k]
                if q is None: continue
                if v.units != q.units:
                    kwargs[k] = v.rescale(q.units)
        except (AttributeError, ValueError):
            try:
                us = str(v.dimensionality)
            except AttributeError:
                us = 'dimensionless'
            raise ValueError(
                'Cannot convert between units of "%s" and "%s"' %
                (us, str(q.dimensionality))
                )
        return f(*args, **kwargs)
    wrapped.args = q_args
    wrapped.kwargs = q_kwargs
    return wrapped


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

##########################     LICENCE     ###############################
##
##   Copyright (c) 2005, Michele Simionato
##   All rights reserved.
##
##   Redistributions of source code must retain the above copyright 
##   notice, this list of conditions and the following disclaimer.
##   Redistributions in bytecode form must reproduce the above copyright
##   notice, this list of conditions and the following disclaimer in
##   the documentation and/or other materials provided with the
##   distribution. 

##   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
##   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
##   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
##   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
##   HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
##   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
##   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
##   OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
##   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
##   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
##   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
##   DAMAGE.

DEF = re.compile('\s*def\s*([_\w][_\w\d]*)\s*\(')

# basic functionality
class FunctionMaker(object):
    """
    An object with the ability to create functions with a given signature.
    It has attributes name, doc, module, signature, defaults, dict and
    methods update and make.
    """
    def __init__(self, func=None, name=None, signature=None,
                 defaults=None, doc=None, module=None, funcdict=None):
        if func:
            # func can be a class or a callable, but not an instance method
            self.name = func.__name__
            if self.name == '<lambda>': # small hack for lambda functions
                self.name = '_lambda_' 
            self.doc = func.__doc__
            self.module = func.__module__
            if inspect.isfunction(func):
                argspec = inspect.getargspec(func)
                self.args, self.varargs, self.keywords, self.defaults = argspec
                for i, arg in enumerate(self.args):
                    setattr(self, 'arg%d' % i, arg)
                self.signature = inspect.formatargspec(
                    formatvalue=lambda val: "", *argspec)[1:-1]
                self.dict = func.__dict__.copy()
        if name:
            self.name = name
        if signature is not None:
            self.signature = signature
        if defaults:
            self.defaults = defaults
        if doc:
            self.doc = doc
        if module:
            self.module = module
        if funcdict:
            self.dict = funcdict
        # check existence required attributes
        assert hasattr(self, 'name')
        if not hasattr(self, 'signature'):
            raise TypeError('You are decorating a non function: %s' % func)

    def update(self, func, **kw):
        "Update the signature of func with the data in self"
        func.__name__ = self.name
        func.__doc__ = getattr(self, 'doc', None)
        func.__dict__ = getattr(self, 'dict', {})
        func.func_defaults = getattr(self, 'defaults', ())
        callermodule = sys._getframe(3).f_globals.get('__name__', '?')
        func.__module__ = getattr(self, 'module', callermodule)
        func.__dict__.update(kw)

    def make(self, src_templ, evaldict=None, addsource=False, **attrs):
        "Make a new function from a given template and update the signature"
        src = src_templ % vars(self) # expand name and signature
        evaldict = evaldict or {}
        mo = DEF.match(src)
        if mo is None:
            raise SyntaxError('not a valid function template\n%s' % src)
        name = mo.group(1) # extract the function name
        reserved_names = set([name] + [
            arg.strip(' *') for arg in self.signature.split(',')])
        for n, v in evaldict.iteritems():
            if n in reserved_names:
                raise NameError('%s is overridden in\n%s' % (n, src))
        if not src.endswith('\n'): # add a newline just for safety
            src += '\n'
        try:
            code = compile(src, '<string>', 'single')
            exec code in evaldict
        except:
            print >> sys.stderr, 'Error in generated code:'
            print >> sys.stderr, src
            raise
        func = evaldict[name]
        if addsource:
            attrs['__source__'] = src
        self.update(func, **attrs)
        return func

    @classmethod
    def create(cls, obj, body, evaldict, defaults=None,
               doc=None, module=None, addsource=True,**attrs):
        """
        Create a function from the strings name, signature and body.
        evaldict is the evaluation dictionary. If addsource is true an attribute
        __source__ is added to the result. The attributes attrs are added,
        if any.
        """
        if isinstance(obj, str): # "name(signature)"
            name, rest = obj.strip().split('(', 1)
            signature = rest[:-1] #strip a right parens            
            func = None
        else: # a function
            name = None
            signature = None
            func = obj
        fun = cls(func, name, signature, defaults, doc, module)
        ibody = '\n'.join('    ' + line for line in body.splitlines())
        return fun.make('def %(name)s(%(signature)s):\n' + ibody, 
                        evaldict, addsource, **attrs)
  
def decorator(caller, func=None):
    """
    decorator(caller) converts a caller function into a decorator;
    decorator(caller, func) decorates a function using a caller.
    """
    if func is not None: # returns a decorated function
        return FunctionMaker.create(
            func, "return _call_(_func_, %(signature)s)",
            dict(_call_=caller, _func_=func), undecorated=func)
    else: # returns a decorator
        if isinstance(caller, partial):
            return partial(decorator, caller)
        # otherwise assume caller is a function
        f = inspect.getargspec(caller)[0][0] # first arg
        return FunctionMaker.create(
            '%s(%s)' % (caller.__name__, f), 
            'return decorator(_call_, %s)' % f,
            dict(_call_=caller, decorator=decorator), undecorated=caller,
            doc=caller.__doc__, module=caller.__module__)
