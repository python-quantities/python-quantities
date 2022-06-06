from typing import Optional

from quantities.dimensionality import Dimensionality
from quantities.typing.quantities import DimensionalityDescriptor, QuantityData
import numpy.typing as npt

def validate_unit_quantity(value: Quantity) -> Quantity:
    ...

def validate_dimensionality(value: DimensionalityDescriptor) -> Dimensionality:
    ...

def get_conversion_factor(from_u: Quantity, to_u: Quantity) -> float:
    ...


class Quantity(npt.NDArray):

    def __new__(cls, data: QuantityData, units: DimensionalityDescriptor = '',
                dtype: Optional[object] = None, copy: bool = True) -> Quantity:
        ...

    @property
    def dimensionality(self) -> Dimensionality:
        ...

    @property
    def _reference(self) :
        ...

    @property
    def magnitude(self) -> npt.NDArray:
        ...

    @property
    def real(self) -> Quantity:
        ...

    @property
    def imag(self) -> Quantity:
        ...

    @property
    def units(self) -> Quantity:
        ...

    def rescale(self, units: Optional[DimensionalityDescriptor] = None) -> Quantity:
        ...

    def rescale_preferred(self) -> Quantity:
        ...

    def __add__(self, other) -> Quantity:
        ...

    def __radd__(self, other) -> Quantity:
        ...

    def __iadd__(self, other) -> Quantity:
        ...

    def __sub__(self, other) -> Quantity:
        ...

    def __rsub__(self, other) -> Quantity:
        ...

    def __isub__(self, other) -> Quantity:
        ...

    def __mod__(self, other) -> Quantity:
        ...

    def __imod__(self, other) -> Quantity:
        ...

    def __imul__(self, other) -> Quantity:
        ...

    def __rmul__(self, other) -> Quantity:
        ...

    def __itruediv__(self, other) -> Quantity:
        ...

    def __rtruediv__(self, other) -> Quantity:
        ...

    def __pow__(self, power) -> Quantity:
        ...

    def __ipow__(self, other) -> Quantity:
        ...

    def __round__(self, decimals: int = 0) -> Quantity:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __getitem__(self, item: int) -> Quantity:
        ...

    def __setitem__(self, key: int, value: QuantityData) -> None:
        ...

