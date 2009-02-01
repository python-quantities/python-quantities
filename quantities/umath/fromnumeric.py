from __future__ import absolute_import

import numpy
from ..quantities import Quantity
from ..utilities import usedoc


__all__ = ['round', 'around', 'round_']


round = numpy.round
around = numpy.around
round_ = numpy.around
