"""
"""

from quantities.units.unitquantity import UnitCurrent, \
    UnitLuminousIntensity, UnitQuantity
from quantities.units.time import s
from quantities.units.length import m
from quantities.units.energy import J


A = amp = amps = ampere = amperes = \
    UnitCurrent('A')
mA = milliamp = milliamps = \
    UnitCurrent('mA', A/1000)
uA = microamp = microamps = \
    UnitCurrent('uA', mA/1000)
nA = nanoamp = nanoamps = \
    UnitCurrent('nA', uA/1000)
pA = picoamp = picoamps = \
    UnitCurrent('pA', nA/1000)
abampere = \
    UnitCurrent('abampere', 10*A)
gilbert = \
    UnitCurrent('gilbert', 7.957747e-1*A) # TODO: check
statampere = \
    UnitCurrent('statampere', 3.335640e-10*A) # TODO: check
biot = \
    UnitCurrent('biot', 10*A)

cd = candle = candles = candela = candelas = \
    UnitLuminousIntensity('cd')

C = coulomb = \
    UnitQuantity('C', A*s)
V = volt = \
    UnitQuantity('V', J/s/A)
F = farad = \
    UnitQuantity('F', C/V)
ohm = ohms = \
    UnitQuantity('ohm', V/A)
S = siemens = \
    UnitQuantity('S', A/V)
Wb = weber = webers = \
    UnitQuantity('Wb', V*s)
T = tesla = teslas = \
    UnitQuantity('T', Wb/m**2)
H = henry = henrys = \
    UnitQuantity('H', Wb/A)
abfarad = abfarads = \
    UnitQuantity('abfarad', 1e9*farad)
abhenry = abhenry = \
    UnitQuantity('abhenry', 1e-9*henry)
abmho = abmhos = \
    UnitQuantity('abmho', 1e9*S)
abohm = abohms = \
    UnitQuantity('abohm', 1e-9*ohm)
abvolt = abvolts = \
    UnitQuantity('abvolt', 1e-8*V)
e = UnitQuantity('e', 1.60217733-19*C)
chemical_faraday = chemical_faradays = \
    UnitQuantity('chemical_faraday', 9.64957e4*C)
physical_faraday = physical_faradays = \
    UnitQuantity('physical_faraday', 9.65219e4*C)
faraday = faradays = C12_faraday = C12_faradays = \
    UnitQuantity('faraday', 9.648531e4*C)
gamma = gammas = \
    UnitQuantity('gamma', 1e-9*T)
gauss = \
    UnitQuantity('gauss', 1e-4*T)
maxwell = maxwells = \
    UnitQuantity('maxwell', 1e-8*Wb)
Oe = oersted = oersteds = \
    UnitQuantity('Oe', 7.957747e1*A/m)
statcoulomb = statcoulombs = \
    UnitQuantity('statcoulomb', 3.335640e-10*C)
statfarad = statfarads = \
    UnitQuantity('statfarad', 1.112650e-12*F)
stathenry = stathenrys = \
    UnitQuantity('stathenry', 8.987554e11*H)
statmho = statmhos = \
    UnitQuantity('statmho', 1.112650e-12*S)
statohm = statohms = \
    UnitQuantity('statohm', 8.987554e11*ohm)
statvolt = statvolts = \
    UnitQuantity('statvolt', 2.997925e2*V)
unit_pole = unit_poles = \
    UnitQuantity('unit_pole', 1.256637e-7*Wb)

del UnitQuantity, s, m, J
