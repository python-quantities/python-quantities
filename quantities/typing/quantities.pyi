from typing import Union, Iterable

from quantities import Quantity
from quantities.dimensionality import Dimensionality
import numpy as np
import numpy.typing as npt

DimensionalityDescriptor = Union[str, Quantity, Dimensionality]
QuantityData = Union[Quantity, npt.NDArray[Union[np.floating, np.integer]], Iterable[Union[float, int]], float, int]
