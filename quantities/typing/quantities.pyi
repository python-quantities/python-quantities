from typing import Iterable, Union

import numpy as np
import numpy.typing as npt

from quantities import Quantity
from quantities.dimensionality import Dimensionality

DimensionalityDescriptor = Union[str, Quantity, Dimensionality]
QuantityData = Union[Quantity, npt.NDArray[Union[np.floating, np.integer]], Iterable[Union[float, int]], float, int]
