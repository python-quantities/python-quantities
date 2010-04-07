from __future__ import absolute_import

from ._codata import physical_constants
from quantities.quantity import Quantity
from quantities.uncertainquantity import UncertainQuantity


def _cd(name):
    entry = physical_constants[name]
    if False: #entry['precision']:
        return UncertainQuantity(
            entry['value'], entry['units'], entry['precision']
        )
    else:
        return Quantity(entry['value'], entry['units'])
