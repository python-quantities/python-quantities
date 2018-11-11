# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitCurrent, UnitLuminousIntensity, UnitQuantity
from .time import s
from .length import cm, m
from .energy import J, erg
from .velocity import c
from .force import N
from math import pi


A = amp = amps = ampere = amperes = UnitCurrent(
    'ampere',
    symbol='A',
    aliases=['amp', 'amps', 'amperes']
)
mA = milliamp = milliampere = UnitCurrent(
    'milliampere',
    A/1000,
    symbol='mA',
    aliases=['milliamp', 'milliamps', 'milliamperes']
)
uA = microampere = UnitCurrent(
    'microampere',
    mA/1000,
    symbol='uA',
    u_symbol='µA',
    aliases=['microamp', 'microamps', 'microamperes'])
nA = nanoamp = nanoampere = UnitCurrent(
    'nanoampere',
    uA/1000,
    symbol='nA',
    aliases=['nanoamp', 'nanoamps', 'nanoamperes']
)
pA = picoamp = picoampere = UnitCurrent(
    'picoampere',
    nA/1000,
    symbol='pA',
    aliases=['picoamp', 'picoamps', 'picoamperes']
)
aA = abampere = biot = UnitCurrent(
    'abampere',
    10*A,
    symbol='aA',
    aliases=['abamperes', 'biot', 'biots']
)

esu = statcoulomb = statC = franklin = Fr = UnitQuantity(
    'statcoulomb',
    1 * erg**0.5 * cm**0.5,
    symbol='esu',
    aliases=['statcoulombs', 'statC', 'franklin', 'franklins', 'Fr']
)
esu_per_second = statampere = UnitCurrent(
    'statampere',
    esu/s,
    symbol='(esu/s)',
    aliases=['statamperes']
)

ampere_turn = UnitQuantity(
    'ampere_turn',
    1*A
)
Gi = gilbert = UnitQuantity(
    'gilbert',
    10/(4*pi)*ampere_turn,
    symbol='Gi'
)

C = coulomb = UnitQuantity(
    'coulomb',
    A*s,
    symbol='C'
)
mC = millicoulomb = UnitQuantity(
    'millicoulomb',
    1e-3*C,
    symbol='mC'
)
uC = microcoulomb = UnitQuantity(
    'microcoulomb',
    1e-6*C,
    symbol='uC',
    u_symbol='μC'
)
V = volt = UnitQuantity(
    'volt',
    J/C,
    symbol='V',
    aliases=['volts']
)
kV = kilovolt = UnitQuantity(
    'kilovolt',
    1000*V,
    symbol='kV',
    aliases=['kilovolts']
)
mV = millivolt = UnitQuantity(
    'millivolt',
    V/1000,
    symbol='mV',
    aliases=['millivolts']
)
uV = microvolt = UnitQuantity(
    'microvolt',
    V/1e6,
    symbol='uV',
    u_symbol='μV',
    aliases=['microvolts']
)
F = farad = UnitQuantity(
    'farad',
    C/V,
    symbol='F',
    aliases=['farads']
)
mF = UnitQuantity(
    'millifarad',
    F/1000,
    symbol='mF'
)
uF = UnitQuantity(
    'microfarad',
    mF/1000,
    symbol='uF',
    u_symbol='μF'
)
nF = UnitQuantity(
    'nanofarad',
    uF/1000,
    symbol='nF'
)
pF = UnitQuantity(
    'picofarad',
    nF/1000,
    symbol='pF'
)
ohm = Ohm = UnitQuantity(
    'ohm',
    V/A,
    u_symbol='Ω',
    aliases=['ohms', 'Ohm']
)
kOhm = UnitQuantity(
    'kiloohm',
    ohm*1000,
    u_symbol='kΩ',
    aliases=['kOhm', 'kohm', 'kiloohms']
)
MOhm = UnitQuantity(
    'megaohm',
    kOhm*1000,
    u_symbol='MΩ',
    aliases=['MOhm', 'Mohm', 'megaohms']
)
S = siemens = UnitQuantity(
    'siemens',
    A/V,
    symbol='S'
)
mS = millisiemens = UnitQuantity(
    'millisiemens',
    S/1000,
    symbol='mS'
)
uS = microsiemens = UnitQuantity(
    'microsiemens',
    mS/1000,
    symbol='uS',
    u_symbol='μS'
)
nS = nanosiemens = UnitQuantity(
    'nanosiemens',
    uS/1000,
    symbol='nS'
)
pS = picosiemens = UnitQuantity(
    'picosiemens',
    nS/1000,
    symbol='pS'
)
Wb = weber = UnitQuantity(
    'weber',
    V*s,
    symbol='Wb',
    aliases=['webers']
)
T = tesla = UnitQuantity(
    'tesla',
    Wb/m**2,
    symbol='T',
    aliases=['teslas']
)
H = henry = UnitQuantity(
    'henry',
    Wb/A,
    symbol='H'
)
abfarad = UnitQuantity(
    'abfarad',
    1e9*farad,
    aliases=['abfarads']
)
abhenry = UnitQuantity(
    'abhenry',
    1e-9*henry
)
abmho = UnitQuantity(
    'abmho',
    1e9*S
)
abohm = UnitQuantity(
    'abohm',
    1e-9*ohm
)
abvolt = UnitQuantity(
    'abvolt',
    1e-8*V,
    aliases=['abvolts']
)
e = elementary_charge = UnitQuantity(
    'elementary_charge',
    1.602176487e-19*C,
    symbol='e',
    doc='relative uncertainty = 6.64e-8'
)
chemical_faraday = UnitQuantity(
    'chemical_faraday',
    9.64957e4*C
)
physical_faraday = UnitQuantity(
    'physical_faraday',
    9.65219e4*C
)
faraday = C12_faraday = UnitQuantity(
    'faraday',
    96485.3399*C,
    aliases=['faradays'],
    doc='The symbol F is reserved for the farad'
)
gamma = UnitQuantity(
    'gamma',
    1e-9*T
)
gauss = UnitQuantity(
    'gauss',
    1e-4*T,
    symbol='G'
)
maxwell = UnitQuantity(
    'maxwell',
    1e-8*Wb,
    symbol='Mx',
    aliases=['maxwells']
)
Oe = oersted = UnitQuantity(
    'oersted',
    1000/(4*pi)*A/m,
    symbol='Oe',
    aliases=['aliases']
)
statfarad = statF = stF = UnitQuantity(
    'statfarad',
    1.112650e-12*F,
    symbol='stF',
    aliases=['statfarads', 'statF']
)
stathenry = statH = stH = UnitQuantity(
    'stathenry',
    8.987554e11*H,
    symbol='stH',
    aliases=['statH']
)
statmho = statS = stS = UnitQuantity(
    'statmho',
    1.112650e-12*S,
    symbol='stS'
)
statohm = UnitQuantity(
    'statohm',
    8.987554e11*ohm,
    u_symbol='stΩ',
    aliases=['statohms']
)
statvolt = statV = stV = UnitQuantity(
    'statvolt',
    2.997925e2*V,
    symbol='stV',
    aliases=['statvolts', 'statV']
)
unit_pole = UnitQuantity(
    'unit_pole',
    1.256637e-7*Wb
)
vacuum_permeability = mu_0 = magnetic_constant = UnitQuantity(
    'magnetic_constant',
    4*pi*10**-7*N/A**2,
    symbol='mu_0',
    u_symbol='μ₀',
    aliases=['vacuum_permeability']
)
vacuum_permittivity = epsilon_0 = electric_constant = UnitQuantity(
    'electric_constant',
    1/(mu_0*c**2),
    symbol='epsilon_0',
    u_symbol='ε₀',
    aliases=['vacuum_permittivity']
)
Z_0 = impedence_of_free_space = characteristic_impedance_of_vacuum = \
        UnitQuantity(
    'characteristic_impedance_of_vacuum',
    mu_0*c,
    symbol='Z_0',
    u_symbol='Z₀',
    aliases=['impedence_of_free_space']
)

cd = candle = candela = UnitLuminousIntensity(
    'candela',
    symbol='cd',
    aliases=['candle', 'candles', 'candelas']
)

del UnitQuantity, s, m, J, c
