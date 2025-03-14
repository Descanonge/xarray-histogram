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
from tests.util import bool_param, get_array, get_ref_hist, get_weights, id_data, id_x


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


class TestStorage:
    def test_storage_warning(self):
        with pytest.warns(UserWarning, match="^Accumulator storages are not supported"):
            xh.histogram(get_array([5]), range=(0, 1), storage=bh.storage.Weight())

    def test_storage_dtype(self):
        x = get_array([10])
        # Default
        hist = xh.histogram(x, range=(0, 1))
        assert np.isdtype(hist.dtype, kind="real floating")

        # Int64
        hist = xh.histogram(x, range=(0, 1), storage=bh.storage.Int64())
        assert np.isdtype(hist.dtype, kind="signed integer")


class TestCoordinates:
    def test_coordinates_float(self):
        ax = bh.axis.Regular(30, 0.0, 1.0)
        x = get_array([20])
        hist = xh.histogram(x, ax)

        coord = hist["var1_bins"]
        assert coord.size == ax.size
        assert coord.dtype == x.dtype
        assert coord.attrs["bin_type"] == "Regular"
        assert coord.attrs["right_edge"] == 1.0
        assert coord.attrs["underflow"] is False
        assert coord.attrs["overflow"] is False
        assert_allclose(coord, ax.edges[:-1])

        # with flow
        hist = xh.histogram(x, ax, flow=True)

        coord = hist["var1_bins"]
        assert coord.size == ax.extent
        assert coord.dtype == x.dtype
        assert coord.attrs["bin_type"] == "Regular"
        assert coord.attrs["right_edge"] == 1.0
        assert coord.attrs["underflow"] is True
        assert coord.attrs["overflow"] is True
        assert_allclose(coord[1:-1], ax.edges[:-1])
        assert coord[0] == -np.inf
        assert coord[-1] == np.inf

    def test_coordinate_int(self):
        ax = bh.axis.Integer(0, 5)
        dt = np.dtype("int16")
        x = get_array([20]).astype(dt)
        hist = xh.histogram(x, ax)

        coord = hist["var1_bins"]
        assert coord.size == ax.size
        assert coord.dtype == dt
        assert coord.attrs["bin_type"] == "Integer"
        assert coord.attrs["underflow"] is False
        assert coord.attrs["overflow"] is False
        assert_allclose(coord, ax.edges[:-1])

        # with flow
        hist = xh.histogram(x, ax, flow=True)

        coord = hist["var1_bins"]
        assert coord.size == ax.extent
        assert coord.dtype == dt
        assert coord.attrs["bin_type"] == "Integer"
        assert coord.attrs["underflow"] is True
        assert coord.attrs["overflow"] is True
        assert_allclose(coord[1:-1], ax.edges[:-1])
        assert coord[0] == np.iinfo(dt).min
        assert coord[-1] == np.iinfo(dt).max

    def test_coordinate_uint(self):
        ax = bh.axis.Integer(0, 5)
        dt = np.dtype("uint16")
        x = get_array([20]).astype(dt)
        hist = xh.histogram(x, ax)

        coord = hist["var1_bins"]
        assert coord.size == ax.size
        assert coord.dtype == dt
        assert coord.attrs["bin_type"] == "Integer"
        assert coord.attrs["underflow"] is False
        assert coord.attrs["overflow"] is False
        assert_allclose(coord, ax.edges[:-1])

        # with flow
        hist = xh.histogram(x, ax, flow=True)
        dt_target = np.dtype("int32")

        coord = hist["var1_bins"]
        assert coord.size == ax.extent
        assert coord.dtype == dt_target
        assert coord.attrs["bin_type"] == "Integer"
        assert coord.attrs["underflow"] is True
        assert coord.attrs["overflow"] is True
        assert_allclose(coord[1:-1], ax.edges[:-1])
        assert coord[-1] == np.iinfo(dt_target).max

    def test_coordinate_int_category(self):
        ax = bh.axis.IntCategory([2, 5, 8, 7])
        x = get_array([20]).astype("int")
        hist = xh.histogram(x, ax)

        coord = hist["var1_bins"]
        assert coord.size == ax.size
        assert np.isdtype(coord.dtype, kind="signed integer")
        assert coord.attrs["bin_type"] == "IntCategory"
        assert coord.attrs["underflow"] is False
        assert coord.attrs["overflow"] is False
        assert_allclose(coord, [2, 5, 8, 7])

        # with flow
        hist = xh.histogram(x, ax, flow=True)

        coord = hist["var1_bins"]
        assert coord.size == ax.extent
        assert np.isdtype(coord.dtype, kind="signed integer")
        assert coord.attrs["bin_type"] == "IntCategory"
        assert coord.attrs["underflow"] is False
        assert coord.attrs["overflow"] is True
        assert_allclose(coord[:-1], [2, 5, 8, 7])
        assert coord[-1] == np.iinfo(x.dtype).max


class TestUnivariate:
    """Check histogram of a single variable."""

    @pytest.mark.parametrize(
        "x",
        [
            get_array([20]),
            get_array([20], chunks=[-1]),
            get_array([20], chunks=[5]),
            get_array([4, 8]),
            get_array([4, 8], chunks=[1, 8]),
            get_array([4, 8], chunks=[4, 2]),
            get_array([4, 8], chunks=[2, 2]),
        ],
        ids=id_x,
    )
    @bool_param("weight")
    @bool_param("density")
    def test_flat(self, x: xr.DataArray, weight: bool, density: bool):
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
        else:
            assert isinstance(hist.data, np.ndarray)

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
        else:
            assert isinstance(hist.data, np.ndarray)

        atol = 0 if density else 1
        for i in range(x.sizes["x"]):
            w = weights.isel(x=i) if weights is not None else None
            ref = get_ref_hist(x.isel(x=i), axes=[ax], density=density, weights=w)
            assert_allclose(hist.isel(x=i).to_numpy(), ref, atol=atol, rtol=1e-6)

    @pytest.mark.parametrize(
        "x",
        [
            get_array([4, 5, 8]),
            get_array([4, 5, 20], chunks=[1, 1, -1]),
            get_array([4, 5, 20], chunks=[2, 1, 4]),
            get_array([4, 5, 6, 6]),
            get_array([4, 5, 6, 6], chunks=[2, 1, 2, 6]),
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
        else:
            assert isinstance(hist.data, np.ndarray)

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
            get_array([3, 20], chunks=[1, 20]),
            get_array([3, 6, 6]),
            get_array([3, 6, 6], chunks=[1, 2, 6]),
            get_array([3, 6, 6], chunks=[3, 2, 1]),
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
            get_array([3, 6, 6]),
            get_array([3, 6, 6], chunks=[1, 2, 6]),
            get_array([3, 6, 6], chunks=[3, 2, 1]),
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

    def test_weights_partial_dask(self):
        # only weight is dask
        x = get_array([4, 4], chunks=[1, 4])
        weights = get_weights(x, True)
        x = x.compute()
        hist = xh.histogram(x, range=(0, 1), weights=weights)
        assert isinstance(hist.data, da.Array)

        # only data is dask
        x = get_array([4, 4], chunks=[1, 4])
        weights = get_weights(x, True)
        weights = weights.compute()
        hist = xh.histogram(x, range=(0, 1), weights=weights)
        assert isinstance(hist.data, da.Array)

    def test_dask_layers(self):
        x = get_array([8, 8], chunks=[2, 4])
        hist = xh.histogram(x, bins=10, range=(0, 1), dims=["y"])
        h = hist.data

        n_blocks = 4  # 8 / 2

        # Check blocksizes
        assert h.shape == (8, 10)
        assert h.npartitions == n_blocks
        assert h.numblocks == (n_blocks, 1)
        assert h.chunks == ((2, 2, 2, 2), (10,))

        # Check layers
        assert h.name.startswith("sum-hist-aggregate-")
        layers = h.dask._toposort_layers()
        assert len(layers) == 3
        assert layers[0] == x.data.name
        assert layers[1].startswith("hist-on-block-")
        assert layers[2].startswith("sum-hist-aggregate-")


class TestMultivariate:
    """Check ND-histogram over the whole flattened array."""

    @pytest.mark.parametrize("nvar", [2, 3], ids=lambda i: f"nvar{i}")
    @pytest.mark.parametrize(
        "x",
        [
            get_array([20]),
            get_array([20], chunks=[-1]),
            get_array([20], chunks=[5]),
            get_array([4, 8]),
            get_array([4, 8], chunks=[1, 8]),
            get_array([4, 8], chunks=[2, 2]),
            get_array([4, 8], chunks=[4, 8]),
        ],
        ids=id_x,
    )
    @bool_param("weight")
    @bool_param("density")
    def test_multivar_simple(
        self, nvar: int, x: xr.DataArray, weight: bool, density: bool
    ):
        data = [x.rename(f"var{i}") for i in range(nvar)]
        weights = get_weights(x, weight)

        axes = [bh.axis.Regular(30, 0.0, 1.0) for _ in range(nvar)]

        hist = xh.histogramdd(*data, bins=axes, weights=weights, density=density)
        ref = get_ref_hist(*data, axes=axes, weights=weights, density=density)

        atol = 0 if density else 1
        assert_allclose(hist.to_numpy(), ref, atol=atol, rtol=1e-6)

        assert hist.dims == tuple(f"{x.name}_bins" for x in data)

    @pytest.mark.parametrize(
        "data",
        [
            [get_array([20, 2]), get_array([20])],
            [get_array([20, 2], chunks=[5, 1]), get_array([20], chunks=[5])],
            [get_array([4, 8, 2]), get_array([4, 8])],
            [get_array([4, 8, 2], chunks=[2, 2, 1]), get_array([4, 8], chunks=[2, 2])],
        ],
        ids=id_data,
    )
    @bool_param("weight")
    def test_broadcast(self, data: Sequence[xr.DataArray], weight: bool):
        """Test broadcasting of two variables.

        x is [N, repeat] with repeat == 2
        y is [N]
        weights are [N, repeat]
        """
        x, y = [d.rename(f"var{i}") for i, d in enumerate(data)]
        weights = get_weights(x, weight)

        axes = [bh.axis.Regular(30, 0.0, 1.0) for _ in data]

        hist = xh.histogramdd(x, y, bins=axes, weights=weights)

        dim = x.dims[-1]
        y_ref = y.expand_dims({dim: x.sizes[dim]}, axis=-1)
        ref = get_ref_hist(x, y_ref, axes=axes, weights=weights)

        assert_allclose(hist.to_numpy(), ref, atol=1, rtol=1e-6)

    def test_partial_dask(self):
        x = get_array([20], name="var1")
        y = get_array([20], chunks=[5], name="var2")

        axes = [bh.axis.Regular(30, 0.0, 1.0) for _ in range(2)]

        hist = xh.histogramdd(x, y, bins=axes)
        ref = get_ref_hist(x, y, axes=axes)

        assert_allclose(hist.to_numpy(), ref, atol=1, rtol=1e-6)


class TestFlowBins:
    pass
