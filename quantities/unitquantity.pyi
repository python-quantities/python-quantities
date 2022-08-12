from typing import Optional, Union, List

from quantities import Quantity
from quantities.dimensionality import Dimensionality


class UnitQuantity(Quantity):

    def __new__(
            cls, name: str, definition: Optional[Union[Quantity, float, int]] = None, symbol: Optional[str] = None,
            u_symbol: Optional[str] = None,
            aliases: List[str] = [], doc=None
    ) -> UnitQuantity:
        ...

    def __init__(
            self, name: str, definition: Optional[Union[Quantity, float, int]] = None, symbol: Optional[str] = None,
            u_symbol: Optional[str] = None,
            aliases: List[str] = [], doc=None
    ) -> None:
        ...

    def __hash__(self) -> int:
        ...

    @property
    def _reference(self) -> UnitQuantity:
        ...

    @property
    def _dimensionality(self) -> Dimensionality:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def symbol(self) -> str:
        ...

    @property
    def u_symbol(self) -> str:
        ...

    @property
    def units(self) -> UnitQuantity:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __add__(self, other) -> Quantity:
        ...

    def __radd__(self, other) -> Quantity:
        ...

    def __sub__(self, other) -> Quantity:
        ...

    def __rsub__(self, other) -> Quantity:
        ...

    def __mod__(self, other) -> Quantity:
        ...

    def __rmod__(self, other) -> Quantity:
        ...

    def __mul__(self, other) -> Quantity:
        ...

    def __rmul__(self, other) -> Quantity:
        ...

    def __truediv__(self, other) -> Quantity:
        ...

    def __rtruediv__(self, other) -> Quantity:
        ...

    def __pow__(self, other) -> Quantity:
        ...

    def __rpow__(self, other) -> Quantity:
        ...


class Dimensionless(UnitQuantity):

    @property
    def _dimensionality(self) -> Dimensionality:
        ...
