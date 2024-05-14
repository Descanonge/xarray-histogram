"""Test histogram values.

Test 1D and 2D histograms with flattened arrays.
Test 1D and 2D histograms along a dimensions.
Results are compared with those from np.histogram.
Tested arrays are generated automatically. Numpy and Dask arrays are used
"""

import boost_histogram as bh
import dask.array as da
import numpy as np
import pytest
import xarray as xr
import xarray_histogram as xh
from numpy.testing import assert_allclose


class InternalError(Exception):
    pass


np.random.seed(42)


@pytest.mark.parametrize("kind", ["np", "da"])
@pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
@pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
def test_1d_flat(kind, density, weight):
    nbins = 50
    ranges = (-3, 3)
    size = 10_000
    chunk_size = size // 10
    vals = np.random.normal(size=(size)).astype(np.float32)

    if weight:
        weights = np.random.uniform(size=vals.shape).astype(np.float32)
    else:
        weights = None

    answer, _ = np.histogram(
        vals, bins=nbins, range=ranges, density=density, weights=weights
    )

    if kind == "np":
        pass
    elif kind == "da":
        vals = da.from_array(vals, chunks=(chunk_size,))
        if weight:
            weights = da.from_array(weights, chunks=(chunk_size,))
    else:
        raise InternalError

    data = xr.DataArray(vals, dims=["x"], name="data")
    if weight:
        weights = xr.DataArray(weights, dims=["x"], name="weights")

    bins = bh.axis.Regular(nbins, *ranges)
    h = xh.histogram(data, bins=[bins], density=density, weight=weights)

    # Check values
    assert_allclose(answer, h.values, rtol=1)

    # Check metadata
    assert h.shape == (nbins,)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["np", "da"])
@pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
@pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
def test_nd_flat(dim, kind, density, weight):
    nbins = [50, 60, 70][:dim]
    ranges = [(-3, 3), (-2.5, 2.5), (-2, 2)][:dim]
    size = 1000
    chunk_size = size // 10
    vals = [np.random.normal(size=(size,)).astype(np.float32) for _ in range(dim)]

    if weight:
        weights = np.random.uniform(size=vals[0].shape).astype(np.float32)
    else:
        weights = None

    answer, _ = np.histogramdd(
        vals, bins=nbins, range=ranges, density=density, weights=weights
    )

    if kind == "np":
        pass
    elif kind == "da":
        vals = [da.from_array(v, chunks=(chunk_size,)) for v in vals]
        if weight:
            weights = da.from_array(weights, chunks=(chunk_size,))
    else:
        raise InternalError

    data = [xr.DataArray(v, dims=["x"], name=f"data_{i}") for i, v in enumerate(vals)]

    if weight:
        weights = xr.DataArray(weights, dims=["x"], name="weights")

    bins = [bh.axis.Regular(n, *r) for n, r in zip(nbins, ranges, strict=False)]
    h = xh.histogram(*data, bins=bins, density=density, weight=weights)

    # Check values
    assert_allclose(answer, h.values, rtol=1)

    # Check metadata
    assert h.shape == tuple(nbins)


@pytest.mark.parametrize("kind", ["np", "da"])
@pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
@pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
def test_1d_along(kind, density, weight):
    nbins = 50
    ranges = (-3, 3)
    shape = (3, 200, 200)
    chunk_size = (1, 200, 100)
    vals = np.random.normal(size=shape).astype(np.float32)

    if weight:
        weights = np.random.uniform(size=shape).astype(np.float32)
    else:
        weights = None

    if kind == "np":
        pass
    elif kind == "da":
        vals = da.from_array(vals, chunks=chunk_size)
        if weight:
            weights = da.from_array(weights, chunks=chunk_size)
    else:
        raise InternalError

    dims = ["t", "x", "y"]
    data = xr.DataArray(vals, dims=dims, name="data")
    if weight:
        weights = xr.DataArray(weights, dims=dims, name="weights")

    bins = bh.axis.Regular(nbins, *ranges)
    h = xh.histogram(
        data, bins=[bins], density=density, weight=weights, dims=["x", "y"]
    )

    # Check values
    for i in range(shape[0]):
        w = weights[i] if weight else None
        answer, _ = np.histogram(
            vals[i], bins=nbins, range=ranges, density=density, weights=w
        )
        assert_allclose(answer, h.isel(t=i).values, rtol=1)
