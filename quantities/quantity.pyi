from typing import Any, Optional

import numpy.typing as npt

from quantities.dimensionality import Dimensionality
from quantities.typing.quantities import DimensionalityDescriptor, QuantityData

def validate_unit_quantity(value: Quantity) -> Quantity:
    ...


def validate_dimensionality(value: DimensionalityDescriptor) -> Dimensionality:
    ...


def get_conversion_factor(from_u: Quantity, to_u: Quantity) -> float:
    ...


def scale_other_units(f: Any) -> None:
    ...


class Quantity(npt.NDArray):

    def __new__(cls, data: QuantityData, units: DimensionalityDescriptor = ...,
                dtype: Optional[object] = ..., copy: bool = ...) -> Quantity:
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

    @property # type: ignore[misc]
    def real(self) -> Quantity:  # type: ignore[override]
        ...

    @property # type: ignore[misc]
    def imag(self) -> Quantity:  # type: ignore[override]
        ...

    @property
    def units(self) -> Quantity:
        ...

    def rescale(self, units: Optional[DimensionalityDescriptor] = ...) -> Quantity:
        ...

    def rescale_preferred(self) -> Quantity:
        ...

    # numeric methods
    def __add__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]
    def __radd__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]
    def __iadd__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]

    def __sub__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]
    def __rsub__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]
    def __isub__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]

    def __mul__(self, other) -> Quantity: ...
    def __rmul__(self, other) -> Quantity: ...
    def __imul__(self, other) -> Quantity: ...

    # NOTE matmul is not supported

    def __truediv__(self, other) -> Quantity: ...  # type: ignore[override]
    def __rtruediv__(self, other) -> Quantity: ...  # type: ignore[override]
    def __itruediv__(self, other) -> Quantity: ...  # type: ignore[override]

    def __floordiv__(self, other) -> Quantity: ...  # type: ignore[override]
    def __rfloordiv__(self, other) -> Quantity: ...  # type: ignore[override]
    def __ifloordiv__(self, other) -> Quantity: ...  # type: ignore[override]

    def __mod__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]
    def __rmod__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]
    def __imod__(self, other: Quantity) -> Quantity: ...  # type: ignore[override]

    # NOTE divmod is not supported

    def __pow__(self, power) -> Quantity: ...
    def __rpow__(self, power) -> Quantity: ...
    def __ipow__(self, power) -> Quantity: ...

    # shift and bitwise are not supported

    # unary methods
    def __neg__(self) -> Quantity: ...
    # def __pos__(self) -> Quantity: ...  # GH#94
    def __abs__(self) -> Quantity: ...
    # NOTE invert is not supported

    def __round__(self, decimals: int = ...) -> Quantity:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __getitem__(self, item: Any) -> Quantity:
        ...

    def __setitem__(self, key: int, value: QuantityData) -> None:
        ...
