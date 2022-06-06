from typing import Union, Iterable

from quantities import Quantity
from quantities.dimensionality import Dimensionality
import numpy.typing as npt

DimensionalityDescriptor = Union[str, Quantity, Dimensionality]
QuantityData = Union[Quantity, npt.NDArray[Union[float,int]], Iterable[Union[float,int]], float, int]
