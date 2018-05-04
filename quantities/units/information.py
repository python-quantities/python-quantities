"""
"""
from __future__ import absolute_import

from ..unitquantity import UnitQuantity, UnitInformation, dimensionless
from .time import s

bit = UnitInformation(
    'bit',
    aliases=['bits']
)
B = byte = o = octet = UnitInformation(
    'byte',
    8*bit,
    symbol='B',
    aliases=['bytes', 'o', 'octet', 'octets']
)
kB = kilobyte = ko = UnitInformation(
    'kilobyte',
    1000 * byte,
    symbol='kB',
    aliases=['kilobytes', 'kilooctet', 'kilooctets']
)
MB = megabyte = Mo = UnitInformation(
    'megabyte',
    1000 * kilobyte,
    symbol='MB',
    aliases=['megabytes', 'megaoctet', 'megaoctets']
)
GB = gigabyte = Go = UnitInformation(
    'gigabyte',
    1000 * megabyte,
    symbol='GB',
    aliases=['gigabytes', 'gigaoctet', 'gigaoctets']
)
TB = terabyte = To = UnitInformation(
    'terabyte',
    1000 * gigabyte,
    symbol='TB',
    aliases=['terabytes', 'teraoctet', 'teraoctets']
)
PB = petabyte = Po = UnitInformation(
    'petabyte',
    1000 * terabyte,
    symbol='PB',
    aliases=['petabytes', 'petaoctet', 'petaoctets']
)
EB = exabyte = Eo = UnitInformation(
    'exabyte',
    1000 * petabyte,
    symbol='EB',
    aliases=['exabytes', 'exaoctet', 'exaoctets']
)
ZB = zettabyte = Zo = UnitInformation(
    'zettabyte',
    1000 * exabyte,
    symbol='ZB',
    aliases=['zettabytes', 'zettaoctet', 'zettaoctets']
)
YB = yottabyte = Yo = UnitInformation(
    'yottabyte',
    1000 * zettabyte,
    symbol='YB',
    aliases=['yottabytes', 'yottaoctet', 'yottaoctets']
)
Bd = baud = bps = UnitQuantity(
    'baud',
    bit/s,
    symbol='Bd',
)

# IEC
KiB = kibibyte = Kio = UnitInformation(
    'kibibyte',
    1024 * byte,
    symbol='KiB',
    aliases=['kibibytes', 'kibioctet', 'kibioctets']
)
MiB = mebibyte = Mio = UnitInformation(
    'mebibyte',
    1024 * kibibyte,
    symbol='MiB',
    aliases=['mebibytes', 'mebioctet', 'mebioctets']
)
GiB = gibibyte = Gio = UnitInformation(
    'gibibyte',
    1024 * mebibyte,
    symbol='GiB',
    aliases=['gibibytes', 'gibioctet', 'gibioctets']
)
TiB = tebibyte = Tio = UnitInformation(
    'tebibyte',
    1024 * gibibyte,
    symbol='TiB',
    aliases=['tebibytes', 'tebioctet', 'tebioctets']
)
PiB = pebibyte = Pio = UnitInformation(
    'pebibyte',
    1024 * tebibyte,
    symbol='PiB',
    aliases=['pebibytes', 'pebioctet', 'pebioctets']
)
EiB = exbibyte = Eio = UnitInformation(
    'exbibyte',
    1024 * pebibyte,
    symbol='EiB',
    aliases=['exbibytes', 'exbioctet', 'exbioctets']
)
ZiB = zebibyte = Zio = UnitInformation(
    'zebibyte',
    1024 * exbibyte,
    symbol='ZiB',
    aliases=['zebibytes', 'zebioctet', 'zebioctets']
)
YiB = yobibyte = Yio = UnitInformation(
    'yobibyte',
    1024 * zebibyte,
    symbol='YiB',
    aliases=['yobibytes', 'yobioctet', 'yobioctets']
)

del UnitQuantity, s, dimensionless
