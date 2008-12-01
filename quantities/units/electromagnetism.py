"""
"""

from quantities.units.unitquantities import compound, current, \
    luminous_intensity
from quantities.units.time import s
from quantities.units.length import m
from quantities.units.energy import J


A = amp = amps = ampere = amperes = current('A')
mA = milliamp = milliamps = current('mA')
uA = microamp = microamps = current('uA')
nA = nanoamp = nanoamps = current('nA')
pA = picoamp = picoamps = current('pA')
abampere = current('abampere')
gilbert = current('gilbert')
statampere = current('statampere')
biot = current('biot')

cd = candle = candles = candela = candelas = luminous_intensity('cd')

C = coulomb = A * s
V = volt = J/ s / A
F = farad = C / V
ohm = ohms = V / A
S = siemens = A / V
Wb = weber = webers = V * s
T = tesla = teslas = Wb / m**2
H = henry = henrys = Wb / A
abfarad = abfarads = 1e9 * farad
abhenry = abhenry = 1e-9 * henry
abmho = abmhos = 1e9 * S
abohm = abohms = 1e-9 * ohm
abvolt = abvolts = 1e-8 * V
e = 1.60217733-19 * C
chemical_faraday = chemical_faradays = 9.64957e4 * C
physical_faraday = physical_faradays = 9.65219e4 * C
faraday = faradays = C12_faraday = C12_faradays = 9.648531e4 * C
gamma = gammas = 1e-9 * T
gauss = 1e-4 * T
maxwell = maxwells = 1e-8 * Wb
Oe = oersted = oersteds = 7.957747e1 * A / m
statcoulomb = statcoulombs = 3.335640e-10 * C
statfarad = statfarads = 1.112650e-12 * F
stathenry = stathenrys = 8.987554e11 * H
statmho = statmhos = 1.112650e-12 * S
statohm = statohms = 8.987554e11 * ohm
statvolt = statvolts = 2.997925e2 * V
unit_pole = unit_poles = 1.256637e-7 * Wb

C_ = coulomb_ = compound('C')
V_ = volt_ = compound('V')
F_ = farad_ = compound('F')
ohm_ = ohms_ = compound('ohm')
S_ = siemens_ = compound('S')
Wb_ = weber_ = webers_ = compound('Wb')
T_ = tesla_ = teslas_ = compound('T')
H_ = henry_ = henrys_ = compound('H')
abfarad_ = abfarads_ = compound('abfarad')
abhenry_ = abhenry_ = compound('abhenry')
abmho_ = abmhos_ = compound('abmho')
abohm_ = abohms_ = compound('abohm')
abvolt_ = abvolts_ = compound('abvolt')
e_ = compound('e')
chemical_faraday_ = chemical_faradays_ = compound('chemical_faraday')
physical_faraday_ = physical_faradays_ = compound('physical_faraday')
faraday_ = faradays_ = C12_faraday_ = C12_faradays_ = compound('faraday')
gamma_ = gammas_ = compound('gamma')
gauss_ = compound('gauss')
maxwell_ = maxwells_ = compound('maxwell')
Oe_ = oersted_ = oersteds_ = compound('Oe')
statcoulomb_ = statcoulombs_ = compound('statcoulomb')
statfarad_ = statfarads_ = compound('statfarad')
stathenry_ = stathenrys_ = compound('stathenry')
statmho_ = statmhos_ = compound('statmho')
statohm_ = statohms_ = compound('statohm')
statvolt_ = statvolts_ = compound('statvolt')
unit_pole_ = unit_poles_ = compound('unit_pole')

del compound, s, m, J
