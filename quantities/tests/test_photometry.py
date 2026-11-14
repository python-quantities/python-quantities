import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import quantities as pq


@pytest.mark.parametrize(
    "name, symbol, aliases",
    [
        ("lumen", "lm", ("lumens",)),
        ("lux", "lx", ()),
        ("foot_candle", "fc", ("footcandle", "footcandles", "foot_candles")),
        ("phot", "ph", ("phots",)),
        ("nit", "nt", ("nits",)),
        ("stilb", "sb", ("stilbs",)),
    ],
)
def test_photometric_names(name, symbol, aliases):
    unit = getattr(pq, name)
    assert getattr(pq, symbol) is unit
    assert getattr(pq.units, name) is unit
    assert getattr(pq.units.photometry, name) is unit
    for label in (name, symbol, *aliases):
        value = pq.Quantity([1, 2], label)
        assert value.dimensionality == unit.dimensionality
        assert_array_equal(value.magnitude, [1, 2])


@pytest.mark.parametrize(
    "name, reference, factor",
    [
        ("lumen", "cd", 1),
        ("lux", "cd/m**2", 1),
        ("foot_candle", "cd/m**2", 1 / 0.3048**2),
        ("phot", "cd/m**2", 10000),
        ("nit", "cd/m**2", 1),
        ("stilb", "cd/m**2", 10000),
    ],
)
def test_photometric_conversion(name, reference, factor):
    original = np.array([-2.0, 0.0, 3.0])
    value = pq.Quantity(original, name)
    converted = value.rescale(reference)
    assert_allclose(converted.magnitude, original * factor, rtol=1e-14)
    assert_allclose(converted.rescale(name).magnitude, original, rtol=1e-14)
    assert_allclose(value.simplified.magnitude, original * factor, rtol=1e-14)
    assert value.simplified.dimensionality == pq.Quantity(1, reference).dimensionality
    assert_array_equal(value.magnitude, original)
    with pytest.raises(ValueError):
        value.rescale(pq.W)


def test_illuminance_from_flux_and_area():
    flux = pq.Quantity([800, 1600], "lm")
    area = pq.Quantity([2, 4], "m**2")
    assert_allclose((flux / area).rescale("lx").magnitude, [400, 400])
    assert_allclose((10 * pq.fc).rescale(pq.lux).magnitude, 10 / 0.3048**2)
    assert_allclose((pq.cd * pq.sr).rescale(pq.lumen).magnitude, 1)


def test_luminance_from_intensity_and_area():
    intensity = pq.Quantity([1, 2], "cd")
    area = pq.Quantity(1, "cm**2")
    assert_allclose((intensity / area).rescale("nt").magnitude, [10000, 20000])
    assert_allclose((intensity / area).rescale("sb").magnitude, [1, 2])


def test_existing_litre_and_candela_are_preserved():
    assert pq.cd is pq.units.electromagnetism.candela
    assert_allclose(pq.Quantity(1, "L").rescale(pq.m**3).magnitude, 0.001)
