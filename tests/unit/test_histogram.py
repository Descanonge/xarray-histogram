"""Test histogram values.

Test 1D and 2D histograms with flattened arrays.
Test 1D and 2D histograms along a dimensions.
Results are compared with those from np.histogram.
Tested arrays are generated automatically. Numpy and Dask arrays are used
"""

from collections.abc import Sequence

import boost_histogram as bh
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarray_histogram as xh


class InternalError(Exception):
    pass


np.random.seed(42)

# Univariate
# flat: single function with different input arrays
# along 1d/2d: class with numpy and dask (with != chunking)

# Multivariate
# test simple, numpy and dask arrays
# test with only some variables dask arrays

# Density computation, for each test
# Weight: for each test
# (each test run 4 times)


class TestBinsInput:
    """Check input bins are correctly handled."""

    def test_simple_axes(self):
        axes_ref = (bh.axis.Regular(5, 0.0, 10.0), bh.axis.Integer(0, 10))
        assert axes_ref == xh.core.get_axes_from_specs(axes_ref, None, [None, None])

        axes_same = (bh.axis.Regular(5, 0.0, 10.0), bh.axis.Regular(5, 0.0, 10.0))
        assert axes_same == xh.core.get_axes_from_specs(
            bh.axis.Regular(5, 0.0, 10.0), None, [None, None]
        )

    def test_regular_spec(self):
        axes_ref = (bh.axis.Regular(5, 0.0, 10.0),)
        assert axes_ref == xh.core.get_axes_from_specs(5, [(0.0, 10.0)], [None])
        assert axes_ref == xh.core.get_axes_from_specs([5], [(0.0, 10.0)], [None])

    # Maybe use hypothesis?
    def test_compute_min_max(self):
        axes_ref = (bh.axis.Regular(5, 0.0, 10.0), bh.axis.Regular(8, 20.0, 30.0))
        values = [np.linspace(0.0, 10.0, 20), np.linspace(20.0, 30.0, 20)]

        assert axes_ref == xh.core.get_axes_from_specs([5, 8], None, values)
        assert axes_ref == xh.core.get_axes_from_specs(
            [5, 8], [(None, 10.0), (None, 30.0)], values
        )
        assert axes_ref == xh.core.get_axes_from_specs(
            [5, 8], [(0.0, None), (20.0, 30.0)], values
        )

    def test_wrong(self):
        # Wrong length of spec or range
        with pytest.raises(IndexError):
            xh.core.get_axes_from_specs([5, 8], None, [None])
        with pytest.raises(IndexError):
            xh.core.get_axes_from_specs([5, 8], [(0.0, 10.0)], [None, None])


def get_array(
    shape: Sequence[int], chunks: Sequence[int] | None = None, name: str = "var1"
) -> xr.DataArray:
    if chunks is None:
        data = np.random.rand(*shape)
    else:
        data = da.random.random(shape, chunks=chunks)

    return xr.DataArray(data, dims=list("xywzuv"[: len(shape)]), name=name)


def get_weights(x: xr.DataArray, weight: bool) -> xr.DataArray | None:
    if not weight:
        return None
    chunks = [c[0] for c in x.chunks] if x.chunks else None
    return get_array(x.shape, chunks=chunks, name="weights")


def get_ref_hist(
    *data: xr.DataArray,
    axes: Sequence[bh.axis.Axis],
    weights: xr.DataArray | None = None,
    **kwargs,
) -> np.ndarray:
    arg = [x.data for x in data]
    kwargs["bins"] = [ax.edges for ax in axes]
    kwargs["weights"] = weights.data if weights is not None else None

    module = da if xh.core.is_any_dask(data) else np
    if len(data) == 1:
        func = module.histogram
        arg = arg[0]
        kwargs["bins"] = kwargs["bins"][0]
    else:
        func = module.histogramdd

    ref, _ = func(arg, **kwargs)
    if isinstance(ref, da.Array):
        ref = ref.compute()

    return ref


def bool_param(name: str):
    values = [True, False]
    ids = [f"{name}:{v}" for v in values]
    return pytest.mark.parametrize(name, values, ids=ids)


def id_x(x: xr.DataArray):
    s = f"x{x.shape}"
    if x.chunks is not None:
        s += str([d[0] for d in x.chunks])
    return s


class TestStorage:
    def test_storage_warning(self):
        with pytest.warns(UserWarning, match="^Accumulator storages are not supported"):
            xh.histogram(get_array([5]), storage=bh.storage.Weight())

    def test_int_storage(self):
        pass


class TestUnivariate:
    """Check histogram of a single variable."""

    @pytest.mark.parametrize(
        "x",
        [
            get_array([20]),
            get_array([20], chunks=[-1]),
            get_array([20], chunks=[5]),
            get_array([20, 15]),
            get_array([20, 15], chunks=[1, 5]),
            get_array([20, 15], chunks=[4, 5]),
            get_array([20, 15], chunks=[4, -1]),
        ],
        ids=id_x,
    )
    @bool_param("weight")
    @bool_param("density")
    def test_1d(self, x: xr.DataArray, weight: bool, density: bool):
        ax = bh.axis.Regular(30, 0.0, 1.0)
        weights = get_weights(x, weight)
        ref = get_ref_hist(x, axes=[ax], density=density, weights=weights)

        hist = xh.histogram(x, bins=ax, weights=weights, density=density)

        assert hist.name == "var1_pdf" if density else "var1_histogram"
        assert hist.dims == ("var1_bins",)
        assert hist.shape == (ax.size,)

        # Check lazy computation
        if x.chunks is not None:
            assert isinstance(hist.data, da.Array)

        atol = 0 if density else 1
        assert_allclose(hist.to_numpy(), ref, atol=atol, rtol=1e-6)

    @pytest.mark.parametrize(
        "x",
        [
            get_array([5, 20]),
            get_array([5, 20], chunks=[1, -1]),
            get_array([5, 20], chunks=[1, 4]),
        ],
        ids=id_x,
    )
    @bool_param("weight")
    @bool_param("density")
    def test_along_1d(self, x: xr.DataArray, weight: bool, density: bool):
        ax = bh.axis.Regular(30, 0.0, 1.0)
        weights = get_weights(x, weight)
        hist = xh.histogram(x, bins=ax, weights=weights, density=density, dims=["y"])

        assert hist.name == "var1_pdf" if density else "var1_histogram"
        assert hist.dims == ("x", "var1_bins")
        assert hist.shape == (x.sizes["x"], ax.size)

        # Check lazy computation
        if x.chunks is not None:
            assert isinstance(hist.data, da.Array)

        atol = 0 if density else 1
        for i in range(x.sizes["x"]):
            w = weights.isel(x=i) if weights is not None else None
            ref = get_ref_hist(x.isel(x=i), axes=[ax], density=density, weights=w)
            assert_allclose(hist.isel(x=i).to_numpy(), ref, atol=atol, rtol=1e-6)

    @pytest.mark.parametrize(
        "x",
        [
            get_array([4, 5, 20]),
            get_array([4, 5, 20, 10]),
            get_array([4, 5, 20], chunks=[1, 1, -1]),
            get_array([4, 5, 20], chunks=[2, 1, 4]),
            get_array([4, 5, 20, 10], chunks=[2, 1, 4, -1]),
        ],
        ids=id_x,
    )
    @bool_param("weight")
    @bool_param("density")
    def test_along_2d(self, x: xr.DataArray, weight: bool, density: bool):
        ax = bh.axis.Regular(30, 0.0, 1.0)
        weights = get_weights(x, weight)
        dims = [d for d in x.dims if d not in ("x", "y")]
        hist = xh.histogram(x, bins=ax, weights=weights, density=density, dims=dims)

        assert hist.name == "var1_pdf" if density else "var1_histogram"
        assert hist.dims == ("x", "y", "var1_bins")
        assert hist.shape == (x.sizes["x"], x.sizes["y"], ax.size)

        # Check lazy computation
        if x.chunks is not None:
            assert isinstance(hist.data, da.Array)

        atol = 0 if density else 1
        for ix in range(x.sizes["x"]):
            for iy in range(x.sizes["y"]):
                w = weights.isel(x=ix, y=iy) if weights is not None else None
                x_ = x.isel(x=ix, y=iy)
                h = hist.isel(x=ix, y=iy)
                ref = get_ref_hist(x_, axes=[ax], density=density, weights=w)
                assert_allclose(h.to_numpy(), ref, atol=atol, rtol=1e-6)

    @pytest.mark.parametrize(
        "x",
        [
            get_array([3, 20]),
            get_array([3, 20], chunks=[1, 5]),
            get_array([3, 10, 10]),
            get_array([3, 10, 10], chunks=[1, 5, -1]),
            get_array([3, 10, 10], chunks=[-1, 5, 1]),
        ],
        ids=id_x,
    )
    def test_weight_broadcast(self, x: xr.DataArray):
        ax = bh.axis.Regular(30, 0.0, 1.0)
        weights = get_weights(x, True).isel(x=0)
        weights_ref = weights.expand_dims(x=x.sizes["x"])
        ref = xh.histogram(x, bins=ax, weights=weights_ref, dims=x.dims)
        hist = xh.histogram(x, bins=ax, weights=weights, dims=x.dims)
        assert_allclose(hist.to_numpy(), ref.to_numpy(), atol=0.1, rtol=1e-6)

    @pytest.mark.parametrize(
        "x",
        [
            get_array([3, 10, 10]),
            get_array([3, 10, 10], chunks=[1, 5, -1]),
            get_array([3, 10, 10], chunks=[-1, 5, 1]),
        ],
        ids=id_x,
    )
    def test_weight_broadcast_along(self, x: xr.DataArray):
        ax = bh.axis.Regular(30, 0.0, 1.0)
        weights = get_weights(x, True).isel(x=0)
        hist = xh.histogram(x, bins=ax, weights=weights, dims=["y", "w"])

        for i in range(x.sizes["x"]):
            ref = xh.histogram(x.isel(x=i), bins=ax, weights=weights)
            assert_allclose(
                hist.isel(x=i).to_numpy(), ref.to_numpy(), atol=0.1, rtol=1e-6
            )

    # def test_weight_partial_dask(self):
    #     # only weight is dask
    #     # only data is dask
    #     pass

    # def test_dask_layers(self):
    #     # check size of layers and number of partitions ?
    #     pass


# class TestMultivariate:
#     """Check ND-histogram over the whole flattened array."""

#     @pytest.mark.parametrize("kind", ["np", "da"])
#     @pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
#     @pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
#     def test_simple(self):
#         assert 0

#     @pytest.mark.parametrize("kind", ["np", "da"])
#     def test_broadcast(self):
#         assert 0

#     @pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
#     @pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
#     def test_partial_dask(self):
#         """Only some of inputs are dask."""
#         assert 0


# @pytest.mark.parametrize("kind", ["np", "da"])
# @pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
# @pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
# def test_1d_flat(kind, density, weight):
#     nbins = 50
#     ranges = (-3, 3)
#     size = 10_000
#     chunk_size = size // 10
#     vals = np.random.normal(size=(size)).astype(np.float32)

#     if weight:
#         weights = np.random.uniform(size=vals.shape).astype(np.float32)
#     else:
#         weights = None

#     answer, _ = np.histogram(
#         vals, bins=nbins, range=ranges, density=density, weights=weights
#     )

#     if kind == "np":
#         pass
#     elif kind == "da":
#         vals = da.from_array(vals, chunks=(chunk_size,))
#         if weight:
#             weights = da.from_array(weights, chunks=(chunk_size,))
#     else:
#         raise InternalError

#     data = xr.DataArray(vals, dims=["x"], name="data")
#     if weight:
#         weights = xr.DataArray(weights, dims=["x"], name="weights")

#     bins = bh.axis.Regular(nbins, *ranges)
#     h = xh.histogram(data, bins=[bins], density=density, weights=weights)

#     # Check values
#     assert_allclose(answer, h.values, rtol=1)

#     # Check metadata
#     assert h.shape == (nbins,)


# @pytest.mark.parametrize("dim", [2, 3])
# @pytest.mark.parametrize("kind", ["np", "da"])
# @pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
# @pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
# def test_nd_flat(dim, kind, density, weight):
#     nbins = [50, 60, 70][:dim]
#     ranges = [(-3, 3), (-2.5, 2.5), (-2, 2)][:dim]
#     size = 1000
#     chunk_size = size // 10
#     vals = [np.random.normal(size=(size,)).astype(np.float32) for _ in range(dim)]

#     if weight:
#         weights = np.random.uniform(size=vals[0].shape).astype(np.float32)
#     else:
#         weights = None

#     answer, _ = np.histogramdd(
#         vals, bins=nbins, range=ranges, density=density, weights=weights
#     )

#     if kind == "np":
#         pass
#     elif kind == "da":
#         vals = [da.from_array(v, chunks=(chunk_size,)) for v in vals]
#         if weight:
#             weights = da.from_array(weights, chunks=(chunk_size,))
#     else:
#         raise InternalError

#     data = [xr.DataArray(v, dims=["x"], name=f"data_{i}") for i, v in enumerate(vals)]

#     if weight:
#         weights = xr.DataArray(weights, dims=["x"], name="weights")

#     bins = [bh.axis.Regular(n, *r) for n, r in zip(nbins, ranges, strict=False)]
#     h = xh.histogram(*data, bins=bins, density=density, weights=weights)

#     # Check values
#     assert_allclose(answer, h.values, rtol=1)

#     # Check metadata
#     assert h.shape == tuple(nbins)


# @pytest.mark.parametrize("kind", ["np", "da"])
# @pytest.mark.parametrize("density", [True, False], ids=["densT", "densF"])
# @pytest.mark.parametrize("weight", [True, False], ids=["wT", "wF"])
# def test_1d_along(kind, density, weight):
#     nbins = 50
#     ranges = (-3, 3)
#     shape = (3, 200, 200)
#     chunk_size = (1, 200, 100)
#     vals = np.random.normal(size=shape).astype(np.float32)

#     if weight:
#         weights = np.random.uniform(size=shape).astype(np.float32)
#     else:
#         weights = None

#     if kind == "np":
#         pass
#     elif kind == "da":
#         vals = da.from_array(vals, chunks=chunk_size)
#         if weight:
#             weights = da.from_array(weights, chunks=chunk_size)
#     else:
#         raise InternalError

#     dims = ["t", "x", "y"]
#     data = xr.DataArray(vals, dims=dims, name="data")
#     if weight:
#         weights = xr.DataArray(weights, dims=dims, name="weights")

#     bins = bh.axis.Regular(nbins, *ranges)
#     h = xh.histogram(
#         data, bins=[bins], density=density, weights=weights, dims=["x", "y"]
#     )

#     # Check values
#     for i in range(shape[0]):
#         w = weights[i] if weight else None
#         answer, _ = np.histogram(
#             vals[i], bins=nbins, range=ranges, density=density, weights=w
#         )
#         assert_allclose(answer, h.isel(t=i).values, rtol=1)
