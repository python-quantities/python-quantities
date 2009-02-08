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


def get_object_signature(obj):
    """
    Get the signature from obj
    """
    import inspect
    try:
        sig = inspect.formatargspec(*inspect.getargspec(obj))
    except TypeError, errmsg:
        msg = "Unable to retrieve the signature of %s '%s'\n"\
              "(Initial error message: %s)"
#        warnings.warn(msg % (type(obj),
#                             getattr(obj, '__name__', '???'),
#                             errmsg))
        sig = ''
    return sig


class _frommethod:
    """Define functions from existing MaskedArray methods.

    Parameters
    ----------
        _methodname : string
            Name of the method to transform.

    """
    def __init__(self, methodname):
        self.__name__ = methodname
        self.__doc__ = self.getdoc()
    #
    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        meth = getattr(MaskedArray, self.__name__, None) or\
               getattr(np, self.__name__, None)
        signature = self.__name__ + get_object_signature(meth)
        if meth is not None:
            doc = """    %s\n%s""" % (signature, getattr(meth, '__doc__', None))
            return doc
    #
    def __call__(self, a, *args, **params):
        if isinstance(a, MaskedArray):
            return getattr(a, self.__name__).__call__(*args, **params)
        #FIXME ----
        #As x is not a MaskedArray, we transform it to a ndarray with asarray
        #... and call the corresponding method.
        #Except that sometimes it doesn't work (try reshape([1,2,3,4],(2,2)))
        #we end up with a "SystemError: NULL result without error in PyObject_Call"
        #A dirty trick is then to call the initial numpy function...
        method = getattr(narray(a, copy=False), self.__name__)
        try:
            return method(*args, **params)
        except SystemError:
            return getattr(np,self.__name__).__call__(a, *args, **params)
