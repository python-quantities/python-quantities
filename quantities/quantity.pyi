from typing import Optional, Any

from quantities.dimensionality import Dimensionality
from quantities.typing.quantities import DimensionalityDescriptor, QuantityData
import numpy.typing as npt


def validate_unit_quantity(value: Quantity) -> Quantity:   #type: ignore
    ...


def validate_dimensionality(value: DimensionalityDescriptor) -> Dimensionality:
    ...


def get_conversion_factor(from_u: Quantity, to_u: Quantity) -> float:
    ...


def scale_other_units(f: Any) -> None:
    ...


class Quantity(npt.NDArray):

    def __new__(cls, data: QuantityData, units: DimensionalityDescriptor = '',
                dtype: Optional[object] = None, copy: bool = True) -> Quantity:   #type: ignore
        ...

    @property
    def dimensionality(self) -> Dimensionality:
        ...

    @property
    def _reference(self):
        ...

    @property
    def magnitude(self) -> npt.NDArray:
        ...

    @property   #type: ignore
    def real(self) -> Quantity:   #type: ignore
        ...

    @property   #type: ignore
    def imag(self) -> Quantity:   #type: ignore
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


    def __sub__(self, other) -> Quantity:   #type: ignore
        ...


    def __rsub__(self, other) -> Quantity:   #type: ignore
        ...


    def __isub__(self, other) -> Quantity:   #type: ignore
        ...

    def __mod__(self, other) -> Quantity:
        ...

    def __imod__(self, other) -> Quantity:
        ...

    #  def __imul__(self, other):
    #      ...

    def __rmul__(self, other) -> Quantity:
        ...

    #   def __itruediv__(self, other) :
    #       ...


    def __rtruediv__(self, other) -> Quantity:  #type: ignore
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

    def __getitem__(self, item: Any) -> Quantity:
        ...

    def __setitem__(self, key: int, value: QuantityData) -> None:
        ...
