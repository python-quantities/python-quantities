from typing import Optional, Union, List

from quantities import Quantity
from quantities.dimensionality import Dimensionality


class UnitQuantity(Quantity):
    _primary_order: int
    _secondary_order: int
    _reference_quantity: Optional[Quantity]

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


class IrreducibleUnit(UnitQuantity):
    _default_unit: Optional[UnitQuantity]

    @property
    def simplified(self) -> Quantity:
        ...

    @classmethod
    def get_default_unit(cls) -> Optional[UnitQuantity]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitQuantity]):
        ...


class UnitMass(IrreducibleUnit):
    _default_unit: Optional[UnitMass]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitMass]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitMass]):
        ...


class UnitLength(IrreducibleUnit):
    _default_unit: Optional[UnitLength]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitLength]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitLength]):
        ...


class UnitTime(IrreducibleUnit):
    _default_unit: Optional[UnitTime]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitTime]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitTime]):
        ...


class UnitCurrent(IrreducibleUnit):
    _default_unit: Optional[UnitCurrent]


@classmethod


def get_default_unit(cls) -> Optional[UnitCurrent]:
    ...


@classmethod
def set_default_unit(cls, unit: Union[str, UnitCurrent]):
    ...


class UnitLuminousIntensity(IrreducibleUnit):
    _default_unit: Optional[UnitLuminousIntensity]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitLuminousIntensity]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitLuminousIntensity]):
        ...


class UnitSubstance(IrreducibleUnit):
    _default_unit: Optional[UnitSubstance]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitSubstance]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitSubstance]):
        ...


class UnitTemperature(IrreducibleUnit):
    _default_unit: Optional[UnitTemperature]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitTemperature]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitTemperature]):
        ...


class UnitInformation(IrreducibleUnit):
    _default_unit: Optional[UnitInformation]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitInformation]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitInformation]):
        ...


class UnitCurrency(IrreducibleUnit):
    _default_unit: Optional[UnitCurrency]

    @classmethod
    def get_default_unit(cls) -> Optional[UnitCurrency]:
        ...

    @classmethod
    def set_default_unit(cls, unit: Union[str, UnitCurrency]):
        ...


class CompoundUnit(UnitQuantity):
    ...


class Dimensionless(UnitQuantity):

    @property
    def _dimensionality(self) -> Dimensionality:
        ...


class UnitConstant(UnitQuantity):
    ...


def set_default_units(system: Optional[str], currency: Optional[Union[str, UnitCurrency]],
                      current: Optional[Union[str, UnitCurrent]], information: Optional[Union[str, UnitInformation]],
                      length: Optional[Union[str, UnitLength]],
                      luminous_intensity: Optional[Union[str, UnitLuminousIntensity]],
                      mass: Optional[Union[str, UnitMass]], substance: Optional[Union[str, UnitSubstance]],
                      temperature: Optional[Union[str, UnitTemperature]], time: Optional[Union[str, UnitTime]]):
    ...
