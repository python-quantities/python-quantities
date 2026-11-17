"""
"""

from ..unitquantity import UnitConstant
from ._utils import _cd

weak_mixing_angle = UnitConstant(
    'weak_mixing_angle',
    _cd('weak mixing angle')
)

del UnitConstant, _cd
