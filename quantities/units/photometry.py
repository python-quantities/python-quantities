"""Units of luminous flux, illuminance and luminance."""

from ..unitquantity import UnitQuantity
from .angle import sr
from .electromagnetism import cd
from .length import cm, ft, m

lm = lumen = UnitQuantity(
    'lumen',
    cd*sr,
    symbol='lm',
    aliases=['lumens']
)
lx = lux = UnitQuantity(
    'lux',
    lumen/m**2,
    symbol='lx'
)
fc = foot_candle = footcandle = UnitQuantity(
    'foot_candle',
    lumen/ft**2,
    symbol='fc',
    aliases=['footcandle', 'footcandles', 'foot_candles']
)
ph = phot = UnitQuantity(
    'phot',
    lumen/cm**2,
    symbol='ph',
    aliases=['phots']
)
nt = nit = UnitQuantity(
    'nit',
    cd/m**2,
    symbol='nt',
    aliases=['nits']
)
sb = stilb = UnitQuantity(
    'stilb',
    cd/cm**2,
    symbol='sb',
    aliases=['stilbs']
)

del UnitQuantity, sr, cd, cm, ft, m
