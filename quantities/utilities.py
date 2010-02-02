from functools import wraps


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
