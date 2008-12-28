

class Dimensionality(dict):

    """
    """

    def __repr__(self):
        num = []
        den = []
        for u, d in self.iteritems():
            u = u.units
            if d>0:
                if d != 1: u = u + ('^%s'%d).rstrip('.0')
                num.append(u)
            elif d<0:
                d = -d
                if d != 1: u = u + ('^%s'%d).rstrip('.0')
                den.append(u)
        res = ' * '.join(num)
        if len(den):
            if not res: res = '1'
            res = res + ' / ' + ' '.join(den)
        if not res: res = '(dimensionless)'
        return res


    def __add__(self, other):
        assert self == other
        return Dimensionality(self)

    __sub__ = __add__

    def __mul__(self, other):
        new = Dimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] += power
            except KeyError:
                new[unit] = power
        return new

    def __div__(self, other):
        new = Dimensionality(self)
        for unit, power in other.iteritems():
            try:
                new[unit] -= power
            except KeyError:
                new[unit] = -power
        return new

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        new = Dimensionality(self)
        for i in new:
            new[i] *= other
        return new


class HasDimensions(object):

    def __init__(self, dimensionality=None):
        self._dimensionality = Dimensionality()
        if dimensionality is not None:
            self._dimensionality.update(dimensionality)

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def magnitude(self):
        return 1.0

    def __add__(self, other):
        return self.dimensionality + other.dimensionality

    __sub__ = __add__

    def __mul__(self, other):
        return self.dimensionality * other.dimensionality

    def __div__(self, other):
        return self.dimensionality / other.dimensionality

    def __pow__(self, other):
        return self.dimensionality**other


class ReferenceUnit(HasDimensions):

    def __init__(self, name):
        HasDimensions.__init__(self, {self:1})
        self._name = name

    @property
    def fundamental_units(self):
        return self

    @property
    def units(self):
        try:
            return self._name
        except:
            return ''

    def __repr__(self):
        return self.units

    def __add__(self, other):
        assert isinstance(other, HasDimensions)
        dims = HasDimensions.__add__(self, other)
        magnitude = self.magnitude + other.magnitude
        return Quantity(dims)

    __sub__ = __add__

    def __mul__(self, other):
        assert isinstance(other, (HasDimensions, int, float))
        try:
            dims = HasDimensions.__mul__(self, other)
            magnitude = self.magnitude * other.magnitude
        except:
            dims = self.dimensionality
            magnitude = self.magnitude * other
        return Quantity(magnitude, dims)

    def __div__(self, other):
        assert isinstance(other, (HasDimensions, int, float))
        try:
            dims = HasDimensions.__div__(self, other)
            magnitude = self.magnitude / other.magnitude
        except:
            dims = self.dimensionality
            magnitude = self.magnitude / other
        return Quantity(magnitude, dims)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        dims = HasDimensions.__pow__(self, other)
        return Quantity(self.magnitude**other, dims)


class CompoundUnit(ReferenceUnit):

    def __init__(self, name, units):
        ReferenceUnit.__init__(self, name)
        self._fundamental_units = units

    def __repr__(self):
        return '%g %s'%(self.magnitude, str(self.dimensionality))

    @property
    def fundamental_units(self):
        return self._fundamental_units


class Quantity(CompoundUnit):

    def __init__(self, magnitude, units):
        if isinstance(units, HasDimensions):
            units = units.dimensionality
        assert isinstance(units, Dimensionality)
        HasDimensions.__init__(self, units)

        self._magnitude = magnitude

    @property
    def magnitude(self):
        return self._magnitude

    def __repr__(self):
        return '%g %s'%(self.magnitude, str(self.dimensionality))


m = ReferenceUnit('m')
kg = ReferenceUnit('kg')
s = ReferenceUnit('s')
J = CompoundUnit('J', kg*m**2/s**2)

energy = J*J/J**2

print energy
