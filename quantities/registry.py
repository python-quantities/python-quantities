"""
"""

import copy
import re


class UnitRegistry:

    class __Registry:

        __shared_state = {}

        def __init__(self):
            self.__dict__ = self.__shared_state
            self.__context = {}

        def __getitem__(self, string):
            try:
                return eval(string, self.__context)
            except NameError:
                # could return self['UnitQuantity'](string)
                raise LookupError(
                    f'Unable to parse units: "{string}"'
                )

        def __setitem__(self, string, val):
            assert isinstance(string, str)
            try:
                assert string not in self.__context
            except AssertionError:
                if val == self.__context[string]:
                    return
                raise KeyError(
                    f'{string} has already been registered for {self.__context[string]}'
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
