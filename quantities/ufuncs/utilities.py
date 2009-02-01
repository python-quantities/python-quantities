class usedoc:


    """
    This decorator gives the function the doc string of another method

    doc_method - the method from which to get the doc string
    prefix - a string to append to the front of the doc string
    suffix - a string to append to the beginning of the doc string
    """
    def __init__(self, doc_method, prefix = "", suffix ="" ):


        self.doc = prefix + doc_method.__doc__ + suffix
    def __call__(self, new_method):

        new_method.__doc__ = self.doc
        return new_method
