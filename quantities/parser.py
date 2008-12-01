"""
"""

import copy
import re


class UnableToParseUnits(Exception):

    def __init__(self, label):
        self.label = label
        return

    def __str__(self):
        str = "Label '%s' is not a parseable unit string." % \
              (self.label)
        return str


class UnitRegistry:

    class __Registry:

        __shared_state = {}

        def __init__(self):
            self.__dict__ = self.__shared_state
            self.__context = {}

        def update(self, module):
            d = {}
            for k, v in module.__dict__.items():
                try:
                    # we dont want to be able to modify the units units!
                    if not v.static:
                        v.static = True
                    d[k] = v
                except AttributeError:
                    if not k == '__builtins__': d[k] = v

            self.__context.update(d)

        def __getitem__(self, string):
            return eval(string, self.__context)

        def __setitem__(self, string, val):
            assert isinstance(string, str)
            self.__context[string] = val

    __regex = re.compile(r'([A-Za-z])\.([A-Za-z])')
    __registry = __Registry()

    def __getattr__(self, attr):
        return getattr(self.__registry, attr)

    def __getitem__(self, label):
        """ Parses a string description of a unit e.g., 'g/cc'.
        if suppress_unkown is True and the label cannot be parsed, the returned
        unit is dimensionless otherwise UnableToParseUnits is raised.
        """

        label  = self.__regex.sub(r"\g<1>*\g<2>", label.replace('^', '**'))

        # make sure we can parse the label ....
        if label == "%": label = "percent"
        if label.lower() == "in": label = "inch"
        if label == None or label == "" or label.lower() == 'unitless' \
                or label.lower() == 'unknown':
            label = "dimensionless"

        try:
            _unit = self.__registry[label]
        except UnableToParseUnits:
            try:
                _unit = self.__registry[label.lower()]
            except:
                _unit = self.__registry['compound'](label)

        # this check should be more robust:
        if hasattr(_unit, 'modify_units'):
            return copy.deepcopy(_unit)
        else:
            raise "unrecognized unit: %s"%_unit

    def update(self, module):
        self.__registry.update(module)


unit_registry = UnitRegistry()
