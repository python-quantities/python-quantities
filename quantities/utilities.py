class usedoc:

    """
    This decorator gives the function the doc string of another method

    doc_method - the method from which to get the doc string
    prefix - a string to append to the front of the doc string
    suffix - a string to append to the beginning of the doc string
    """

    def __init__(self, doc_method, prefix=None, suffix=None):
        doc = doc_method.__doc__
        if prefix:
            doc = '\n\n'.join([prefix, doc])
        if suffix:
            note = 'Specific to quantities\n----------------------'
            doc = '\n\n'.join([doc, note, suffix])
        self.doc = doc

    def __call__(self, new_method):
        new_method.__doc__ = self.doc
        return new_method
