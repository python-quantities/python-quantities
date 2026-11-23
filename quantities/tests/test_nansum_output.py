import numpy as np
import pytest

import quantities as pq


@pytest.mark.parametrize('axis', [None, 0, 1])
@pytest.mark.parametrize('units', ['m', 'cm'])
def test_nansum_output_values_and_identity(axis, units):
    data = np.array([[1., np.nan, 3.], [4., 5., np.nan]])
    quantity = pq.Quantity(data, 'm')
    before = quantity.copy()
    expected = np.nansum(data, axis=axis)
    output = pq.Quantity(np.zeros(np.shape(expected)), units)
    result = quantity.nansum(axis=axis, out=output)
    np.testing.assert_allclose(result.rescale('m').magnitude, expected)
    assert result is output
    assert result.dimensionality == pq.m.dimensionality
    np.testing.assert_array_equal(quantity.magnitude, before.magnitude)


def test_nansum_incompatible_output():
    quantity = pq.Quantity([1., np.nan, 3.], 'm')
    output = pq.Quantity(99., 's')
    with pytest.raises(ValueError):
        quantity.nansum(out=output)
    assert output.magnitude == 99.
    assert output.dimensionality == pq.s.dimensionality


@pytest.mark.parametrize('units', ['m', 'cm'])
def test_nansum_output_view(units):
    quantity = pq.Quantity([1., np.nan, 3.], 'm')
    owner = pq.Quantity([99.], units)
    output = owner[0:1].reshape(())
    if units == 'cm':
        with pytest.raises(ValueError, match='view'):
            quantity.nansum(out=output)
        assert owner.magnitude[0] == 99.
    else:
        assert quantity.nansum(out=output) is output
        assert owner.magnitude[0] == 4.
    assert owner.dimensionality == getattr(pq, units).dimensionality


def test_nansum_plain_output_compatibility():
    quantity = pq.Quantity([1., np.nan, 3.], 'm')
    output = np.array(99.)
    result = quantity.nansum(out=output)
    assert result.rescale('m').magnitude == 4.
    assert output == 4.


def test_nansum_output_dtype_and_all_nan():
    quantity = pq.Quantity([np.nan, np.nan], 'm')
    output = pq.Quantity(np.array(99., dtype=np.float32), 'cm', dtype=np.float32)
    result = quantity.nansum(dtype=np.float32, out=output)
    assert result is output
    assert result.dtype == np.float32
    assert result.rescale('m').magnitude == 0.


@pytest.mark.parametrize('data, dtype', [
    (np.array([2 ** 63, 2 ** 63], dtype=np.uint64), np.float64),
    (np.array([1 + 2j, 3 + 4j, complex(np.nan, 0)]), np.complex128),
])
def test_nansum_output_accumulator(data, dtype):
    expected = np.empty((), dtype=dtype)
    np.nansum(data, out=expected)
    output = pq.Quantity(np.empty((), dtype=dtype), 'm', dtype=dtype)
    result = pq.Quantity(data, 'm', dtype=data.dtype).nansum(out=output)
    assert result is output
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result.rescale('m').magnitude, expected)
