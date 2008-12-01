
import copy

from numpy import array, ndarray, zeros

from quantities.quantity import Quantity
from quantities.dimensionality import Dimensionality


def length(units):
    u = ['']*10
    u[1] = units
    dims = zeros(10)
    dims[1] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def mass(units):
    u = ['']*10
    u[0] = units
    dims = zeros(10)
    dims[0] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def time(units):
    u = ['']*10
    u[2] = units
    dims = zeros(10)
    dims[2] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def current(units):
    u = ['']*10
    u[3] = units
    dims = zeros(10)
    dims[3] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def luminous_intensity(units):
    u = ['']*10
    u[4] = units
    dims = zeros(10)
    dims[4] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def substance(units):
    u = ['']*10
    u[5] = units
    dims = zeros(10)
    dims[5] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def temperature(units):
    u = ['']*10
    u[6] = units
    dims = zeros(10)
    dims[6] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def information(units):
    u = ['']*10
    u[7] = units
    dims = zeros(10)
    dims[7] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def angle(units):
    u = ['']*10
    u[8] = units
    dims = zeros(10)
    dims[8] = 1
    return Quantity(1.0, Dimensionality(u, dims), static=True)

def currency(units):
    u = ['']*10
    u[9] = units
    dims = zeros(10)
    dims[9] = 1
    return Quantity(1.0, Dimensionality(u, dims))

def compound(units):
    assert isinstance(units, str)
    if not (units.startswith('(')): units = '(' + units
    if not (units.endswith(')')): units = units + ')'
    u = ['']*10
    dims = zeros(10)
    return Quantity(1.0, Dimensionality(u, dims, units))

#def dimensionless():
#    u = ['']*10
#    dims = zeros(10)
#    return Quantity(1.0, Dimensionality(u, dims))

dimensionless = Quantity(1.0, Dimensionality(['']*10, zeros(10)))
