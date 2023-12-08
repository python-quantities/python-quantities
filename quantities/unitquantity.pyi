from typing import Any, List, Optional, Union, overload

from quantities import Quantity
from quantities.dimensionality import Dimensionality

class UnitQuantity(Quantity):
    _primary_order: int
    _secondary_order: int
    _reference_quantity: Optional[Quantity]

    def __new__(
            cls, name: str, definition: Optional[Union[Quantity, float, int]] = ..., symbol: Optional[str] = ...,
            u_symbol: Optional[str] = ...,
            aliases: List[str] = ..., doc=...
    ) -> UnitQuantity:
        ...

    def __init__(
            self, name: str, definition: Optional[Union[Quantity, float, int]] = ..., symbol: Optional[str] = ...,
            u_symbol: Optional[str] = ...,
            aliases: List[str] = ..., doc=...
    ) -> None:
        ...

    def __hash__(self) -> int:  # type: ignore[override]
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

    def __sub__(self, other) -> Any:
        ...


    def __rsub__(self, other) -> Any:
        ...

    def __mod__(self, other) -> Quantity:
        ...

    def __rmod__(self, other) -> Quantity:
        ...

    def __mul__(self, other) -> Quantity:
        ...

    def __rmul__(self, other) -> Quantity:
        ...

    def __truediv__(self, other) -> Any:
        ...

    def __rtruediv__(self, other) -> Any:
        ...

    def __pow__(self, other) -> Quantity:
        ...

    def __rpow__(self, other) -> Quantity:
        ...


class IrreducibleUnit(UnitQuantity):
    _default_unit: Optional[Quantity]

    @property
    def simplified(self) -> Quantity:
        ...

    @classmethod
    def get_default_unit(cls) -> Optional[Quantity]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, Quantity]):
        ...


class UnitMass(IrreducibleUnit):
    ...


class UnitLength(IrreducibleUnit):
    ...


class UnitTime(IrreducibleUnit):
    ...


class UnitCurrent(IrreducibleUnit):
    ...

class UnitLuminousIntensity(IrreducibleUnit):
    ...


class UnitSubstance(IrreducibleUnit):
    ...


class UnitTemperature(IrreducibleUnit):
    ...


class UnitInformation(IrreducibleUnit):
    ...


class UnitCurrency(IrreducibleUnit):
    ...


class CompoundUnit(UnitQuantity):
    ...


class Dimensionless(UnitQuantity):

    @property
    def _dimensionality(self) -> Dimensionality:
        ...

dimensionless: Dimensionless

class UnitConstant(UnitQuantity):
    ...


def set_default_units(system: Optional[str] = ...,
                      currency: Optional[Union[str, UnitCurrency]] = ...,
                      current: Optional[Union[str, UnitCurrent]] = ...,
                      information: Optional[Union[str, UnitInformation]] = ...,
                      length: Optional[Union[str, UnitLength]] = ...,
                      luminous_intensity: Optional[Union[str, UnitLuminousIntensity]] = ...,
                      mass: Optional[Union[str, UnitMass]] = ...,
                      substance: Optional[Union[str, UnitSubstance]] = ...,
                      temperature: Optional[Union[str, UnitTemperature]] = ...,
                      time: Optional[Union[str, UnitTime]] = ...):
    ...
