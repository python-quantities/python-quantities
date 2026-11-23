import numpy as np
import pytest

import quantities as pq


@pytest.mark.parametrize('entry', ['method', 'numpy'])
@pytest.mark.parametrize('axis, values, unit', [
    ((0,), [3., 8.], pq.m ** 2),
    ((0, 1), 24., pq.m ** 4),
    ((-1,), [2., 12.], pq.m ** 2),
    ((), [[1., 2.], [3., 4.]], pq.m),
])
def test_product_tuple_axes(entry, axis, values, unit):
    quantity = pq.Quantity([[1., 2.], [3., 4.]], 'm')
    before = quantity.copy()
    result = quantity.prod(axis=axis) if entry == 'method' else np.prod(quantity, axis=axis)
    np.testing.assert_allclose(result.rescale(unit).magnitude, values)
    np.testing.assert_array_equal(quantity.magnitude, before.magnitude)
    assert quantity.dimensionality == before.dimensionality


@pytest.mark.parametrize('axis, unit', [((0,), pq.dimensionless), ((), pq.m), ((-1,), pq.m ** 2)])
def test_empty_product_axes(axis, unit):
    data = np.empty((0, 2))
    result = pq.Quantity(data, 'm').prod(axis=axis)
    np.testing.assert_array_equal(result.rescale(unit).magnitude, np.prod(data, axis=axis))


@pytest.mark.parametrize('axis, error', [((0, 0), ValueError), ((2,), np.exceptions.AxisError)])
def test_invalid_product_axes(axis, error):
    quantity = pq.Quantity([[1., 2.], [3., 4.]], 'm')
    with pytest.raises(error):
        quantity.prod(axis=axis)


@pytest.mark.parametrize('axis', [0, -1, ()])
def test_scalar_product_axes(axis):
    result = pq.Quantity(3., 'm').prod(axis=axis)
    assert result.rescale('m').magnitude == 3.


def test_product_axes_output_dtype():
    quantity = pq.Quantity([[1, 2], [3, 4]], 'm', dtype=np.int16)
    output = pq.Quantity(0., 'm**4')
    result = quantity.prod(axis=(0, 1), dtype=np.float64, out=output)
    assert result is output
    assert result.dtype == np.float64
    assert result.rescale('m**4').magnitude == 24.
