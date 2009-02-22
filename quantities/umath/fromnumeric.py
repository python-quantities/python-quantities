from __future__ import absolute_import

import numpy as np

from ..quantity import Quantity
from ..utilities import with_doc


__all__ = ['round', 'around', 'round_']


round = np.round
around = np.around
round_ = np.around
