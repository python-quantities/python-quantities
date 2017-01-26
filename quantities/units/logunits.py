"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitLogIntensity, UnitLogPower, UnitQuantity

bel = UnitLogIntensity(
    'bel',
    symbol='bel'
)
decibel = dB = UnitQuantity(
    'decibel',
    bel / 10,
    'dB'
)

bel_milliwatt = UnitLogPower(
    'bel_milliwatt',
    symbol='Bm'
)
decibel_milliwatt = dBm = UnitQuantity(
    'decibel_milliwatt',
    bel_milliwatt / 10,
    symbol='dBm'
)

del UnitLogIntensity, UnitLogPower, UnitQuantity
