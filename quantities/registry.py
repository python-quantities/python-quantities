"""
"""

import re
import builtins


class UnitRegistry:

    class __Registry:

        __shared_state = {}

        def __init__(self):
            self.__dict__ = self.__shared_state
            self.__context = {}

        def __getitem__(self, string):
            
            # easy hack to prevent arbitrary evaluation of code
            all_builtins = dir(builtins)
            # because we have kilobytes, other bytes we have to remove bytes
            all_builtins.remove("bytes")
            # have to deal with octet as well
            all_builtins.remove("oct")
            # have to remove min which is short for minute
            all_builtins.remove("min")
            for builtin in all_builtins:
                if builtin in string:
                    raise RuntimeError(f"String parsing error for `{string}`. Enter a string accepted by quantities")

            try:
                return eval(string, self.__context)
            except NameError:
                # could return self['UnitQuantity'](string)
                raise LookupError(
                    'Unable to parse units: "%s"'%string
                )

        def __setitem__(self, string, val):
            assert isinstance(string, str)
            try:
                assert string not in self.__context
            except AssertionError:
                if val == self.__context[string]:
                    return
                raise KeyError(
                    '%s has already been registered for %s'
                    % (string, self.__context[string])
                )
            self.__context[string] = val

    __regex = re.compile(r'([A-Za-z])\.([A-Za-z])')
    __registry = __Registry()

    def __getattr__(self, attr):
        return getattr(self.__registry, attr)

    def __setitem__(self, label, value):
        self.__registry.__setitem__(label, value)

    def __getitem__(self, label):
        """Parses a string description of a unit e.g., 'g/cc'"""

        label = self.__regex.sub(
            r"\g<1>*\g<2>", label.replace('^', '**').replace('·', '*'))

        # make sure we can parse the label ....
        if label == '': label = 'dimensionless'
        if "%" in label: label = label.replace("%", "percent")
        if label.lower() == "in": label = "inch"

        return self.__registry[label]

unit_registry = UnitRegistry()
