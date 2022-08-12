from quantities import UnitQuantity


class UnitRegistry:

    def __getitem__(self, item: str) -> UnitQuantity:
        ...

unit_registry: UnitRegistry