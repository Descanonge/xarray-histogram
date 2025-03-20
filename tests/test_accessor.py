#!/usr/bin/env python3

from functools import reduce

import boost_histogram as bh
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarray_histogram as xh
import xarray_histogram.accessor
from tests.util import bool_param, get_array, id_x
from xarray_histogram.accessor import HistDataArrayAccessor


def get_blank_histogram(n_var: int = 1) -> xr.DataArray:
    bins = [f"var{i + 1}_bins" for i in range(n_var)]
    h = xr.DataArray(
        np.empty([10 for i in range(n_var)]),
        dims=bins,
        coords={b: np.arange(0, 10) for b in bins},
        name="_".join([f"var{i + 1}" for i in range(n_var)]) + "_histogram",
    )
    for b in bins:
        h[b].attrs["right_edge"] = 10
        h[b].attrs["bin_type"] = "Regular"

    return h


def get_hist(*axes: bh.axis.Axis, flow=True) -> xr.DataArray:
    data = []
    for i, ax in enumerate(axes):
        x = get_array([2], name=f"var{i + 1}")
        if isinstance(ax, bh.axis.Integer | bh.axis.IntCategory):
            x = x.astype("int")
        if isinstance(ax, bh.axis.StrCategory):
            x = x.astype("U")
        data.append(x)
    return xh.histogramdd(*data, bins=axes, flow=flow)


class TestAccessibility:
    @pytest.mark.parametrize("x", [get_array([20]), get_array([20, 5])], ids=id_x)
    @bool_param("density")
    def test_xh_output_univar(self, x: xr.DataArray, density: bool):
        h = xh.histogram(x, density=density, dims=["x"])

        assert isinstance(h.hist, HistDataArrayAccessor)
        assert h.hist.variables == ["var1"]

    @bool_param("density")
    def test_xh_output_multivar(self, density: bool):
        x, y = get_array([20], name="var1"), get_array([20], name="var2")
        h = xh.histogram2d(x, y, density=density)

        assert isinstance(h.hist, HistDataArrayAccessor)
        assert h.hist.variables == ["var1", "var2"]


def test_variable_argument() -> None:
    # Single variable
    h = get_blank_histogram()
    assert h.hist._variable("var1") == "var1"
    assert h.hist._variable(None) == "var1"

    # Multiple variables
    h = get_blank_histogram(3)
    assert h.hist._variable("var3") == "var3"

    with pytest.raises(ValueError):
        h.hist._variable(None)

    with pytest.raises(KeyError):
        h.hist._variable("bad variable")


class TestCoords:
    def test_edges(self) -> None:
        # Regular
        h = get_hist(bh.axis.Regular(3, 0.0, 0.3), flow=False)
        assert_allclose(h.hist.edges(), [0.0, 0.1, 0.2, 0.3])
        h = get_hist(bh.axis.Regular(3, 0.0, 0.3, underflow=False))
        assert_allclose(h.hist.edges(), [0.0, 0.1, 0.2, 0.3, np.inf])
        h = get_hist(bh.axis.Regular(3, 0.0, 0.3))
        assert_allclose(h.hist.edges(), [-np.inf, 0.0, 0.1, 0.2, 0.3, np.inf])

        # Integer
        h = get_hist(bh.axis.Integer(0, 3), flow=False)
        assert_allclose(h.hist.edges(), [0, 1, 2, 3])
        h = get_hist(bh.axis.Integer(0, 3, underflow=False))
        dtype = h.var1_bins.dtype
        vmin = np.iinfo(dtype).min
        vmax = np.iinfo(dtype).max
        assert_allclose(h.hist.edges(), [0, 1, 2, 3, vmax])
        h = get_hist(bh.axis.Integer(0, 3))
        assert_allclose(h.hist.edges(), [vmin, 0, 1, 2, 3, vmax])

        # Variable
        h = get_hist(bh.axis.Variable([0, 1, 3, 10]), flow=False)
        assert_allclose(h.hist.edges(), [0, 1, 3, 10])
        h = get_hist(bh.axis.Variable([0, 1, 3, 10], underflow=False))
        assert_allclose(h.hist.edges(), [0, 1, 3, 10, np.inf])
        h = get_hist(bh.axis.Variable([0, 1, 3, 10]))
        assert_allclose(h.hist.edges(), [-np.inf, 0, 1, 3, 10, np.inf])

        # Not supported
        for ax in [bh.axis.IntCategory([0, 1, 2]), bh.axis.StrCategory(["a", "b"])]:
            h = get_hist(ax)
            with pytest.raises(TypeError):
                h.hist.edges()

    def test_infer_right_edge(self) -> None:
        h = get_hist(bh.axis.Regular(10, 0.0, 1.0), flow=False)
        # this reset right edge
        h = h.assign_coords(var1_bins=np.arange(0, 10))

        assert isinstance(h.hist, HistDataArrayAccessor)
        assert_allclose(h.hist.edges(), np.arange(0, 11))

        # Unregular edges
        h_wrong = h.assign_coords(var1_bins=np.logspace(1, 10, 10))
        with pytest.raises(ValueError):
            _ = h_wrong.hist

    def test_centers(self):
        # Regular
        h = get_hist(bh.axis.Regular(5, 0.0, 1.0), flow=False)
        assert_allclose(h.hist.centers(), [0.1, 0.3, 0.5, 0.7, 0.9])
        h = get_hist(bh.axis.Regular(5, 0.0, 1.0, underflow=False))
        assert_allclose(h.hist.centers(), [0.1, 0.3, 0.5, 0.7, 0.9, np.inf])
        h = get_hist(bh.axis.Regular(5, 0.0, 1.0))
        assert_allclose(h.hist.centers(), [-np.inf, 0.1, 0.3, 0.5, 0.7, 0.9, np.inf])

        # Variable
        h = get_hist(bh.axis.Variable([0, 1, 3, 9]), flow=False)
        assert_allclose(h.hist.centers(), [0.5, 2, 6])
        h = get_hist(bh.axis.Variable([0, 1, 3, 9], underflow=False))
        assert_allclose(h.hist.centers(), [0.5, 2, 6, np.inf])
        h = get_hist(bh.axis.Variable([0, 1, 3, 9]))
        assert_allclose(h.hist.centers(), [-np.inf, 0.5, 2, 6, np.inf])

        # Integer
        h = get_hist(bh.axis.Integer(0, 4), flow=False)
        assert_allclose(h.hist.centers(), [0.5, 1.5, 2.5, 3.5])
        h = get_hist(bh.axis.Integer(0, 4, underflow=False))
        dtype = h.var1_bins.dtype
        vmin = np.iinfo(dtype).min
        vmax = np.iinfo(dtype).max
        assert_allclose(h.hist.centers(), [0.5, 1.5, 2.5, 3.5, vmax])
        h = get_hist(bh.axis.Integer(0, 4))
        assert_allclose(h.hist.centers(), [vmin, 0.5, 1.5, 2.5, 3.5, vmax])

        # Not supported
        for ax in [bh.axis.IntCategory([0, 1, 2]), bh.axis.StrCategory(["a", "b"])]:
            h = get_hist(ax)
            with pytest.raises(TypeError):
                h.hist.centers()

    def test_widths(self):
        # Regular
        h = get_hist(bh.axis.Regular(5, 0.0, 1.0), flow=False)
        assert_allclose(h.hist.widths(), [0.2, 0.2, 0.2, 0.2, 0.2])
        h = get_hist(bh.axis.Regular(5, 0.0, 1.0, underflow=False))
        assert_allclose(h.hist.widths(), [0.2, 0.2, 0.2, 0.2, 0.2, 1])
        h = get_hist(bh.axis.Regular(5, 0.0, 1.0))
        assert_allclose(h.hist.widths(), [1, 0.2, 0.2, 0.2, 0.2, 0.2, 1])

        # Variable
        h = get_hist(bh.axis.Variable([0, 1, 3, 9]), flow=False)
        assert_allclose(h.hist.widths(), [1, 2, 6])
        h = get_hist(bh.axis.Variable([0, 1, 3, 9], underflow=False))
        assert_allclose(h.hist.widths(), [1, 2, 6, 1])
        h = get_hist(bh.axis.Variable([0, 1, 3, 9]))
        assert_allclose(h.hist.widths(), [1, 1, 2, 6, 1])

        # Integer
        h = get_hist(bh.axis.Integer(0, 4), flow=False)
        assert_allclose(h.hist.widths(), np.ones(4))
        h = get_hist(bh.axis.Integer(0, 4, underflow=False))
        assert_allclose(h.hist.widths(), np.ones(5))
        h = get_hist(bh.axis.Integer(0, 4))
        assert_allclose(h.hist.widths(), np.ones(6))

        # IntCategory
        h = get_hist(bh.axis.IntCategory([1, 3, 5, 8]), flow=False)
        assert_allclose(h.hist.widths(), np.ones(4))
        h = get_hist(bh.axis.IntCategory([1, 3, 5, 8]))
        assert_allclose(h.hist.widths(), np.ones(5))

        # StrCategory
        h = get_hist(bh.axis.StrCategory(["a", "b", "c"]), flow=False)
        assert_allclose(h.hist.widths(), np.ones(3))
        h = get_hist(bh.axis.StrCategory(["a", "b", "c"]))
        assert_allclose(h.hist.widths(), np.ones(4))

    def test_areas(self):
        # the accessor uses functools.reduce(operator.mul) over xr.DataArrays
        # we test against computation done with other numpy functions
        ax1 = bh.axis.Regular(10, 0, 10)
        ax2 = bh.axis.Variable([2, 5, 6, 8, 10])
        ax3 = bh.axis.Integer(5, 10)

        h = get_hist(ax1, ax2, ax3, flow=False)

        # widths
        w1 = np.diff(ax1.edges)
        w2 = np.diff(ax2.edges)
        w3 = np.diff(ax3.edges)

        assert_allclose(h.hist.areas(["var1"]), w1)
        assert_allclose(h.hist.areas(["var1", "var2"]), np.outer(w1, w2))
        assert_allclose(h.hist.areas(["var2", "var3"]), np.outer(w2, w3))
        assert_allclose(h.hist.areas(), reduce(np.multiply, np.ix_(w1, w2, w3)))


class TestTransformBins:
    def test_apply(self):
        # Integer
        h = get_hist(bh.axis.Integer(0, 4), flow=False)
        dtype = h.var1_bins.dtype
        vmin = np.iinfo(dtype).min
        vmax = np.iinfo(dtype).max

        h2 = h.hist.apply_func(lambda bins: bins + 2)
        assert "right_edge" not in h2.hist.bins().attrs
        assert_allclose(h2.hist.bins(), [2, 3, 4, 5])
        assert_allclose(h2.hist.edges(), [2, 3, 4, 5, 6])

        h = get_hist(bh.axis.Integer(0, 4, underflow=False))
        h2 = h.hist.apply_func(lambda bins: bins + 2)
        assert "right_edge" not in h2.hist.bins().attrs
        assert_allclose(h2.hist.bins(), [2, 3, 4, 5, vmax])
        assert_allclose(h2.hist.edges(), [2, 3, 4, 5, 6, vmax])

        h = get_hist(bh.axis.Integer(0, 4))
        h2 = h.hist.apply_func(lambda bins: bins + 2)
        assert "right_edge" not in h2.hist.bins().attrs
        assert_allclose(h2.hist.bins(), [vmin, 2, 3, 4, 5, vmax])
        assert_allclose(h2.hist.edges(), [vmin, 2, 3, 4, 5, 6, vmax])

        # Regular
        h = get_hist(bh.axis.Regular(4, 0, 4), flow=False)

        h2 = h.hist.apply_func(lambda bins: bins + 2)
        assert_allclose(h2.hist.bins(), [2, 3, 4, 5])
        assert h2.hist.bins().attrs["right_edge"] == 6.0
        assert_allclose(h2.hist.edges(), [2, 3, 4, 5, 6])

        h = get_hist(bh.axis.Regular(4, 0, 4, underflow=False))
        h2 = h.hist.apply_func(lambda bins: bins + 2)
        assert h2.hist.bins().attrs["right_edge"] == 6.0
        assert_allclose(h2.hist.edges(), [2, 3, 4, 5, 6, np.inf])

        h = get_hist(bh.axis.Regular(4, 0, 4))
        h2 = h.hist.apply_func(lambda bins: bins + 2)
        assert h2.hist.bins().attrs["right_edge"] == 6.0
        assert_allclose(h2.hist.edges(), [-np.inf, 2, 3, 4, 5, 6, np.inf])

    def test_apply_no_side_effect(self):
        h = get_hist(bh.axis.Regular(10, 10, 20), bh.axis.Integer(0, 10), flow=False)
        h2 = h.hist.apply_func(lambda bins: bins + 2, "var1")
        assert_allclose(h2.hist.edges("var1"), h.hist.edges("var1") + 2)
        assert_allclose(h2.hist.edges("var2"), h.hist.edges("var2"))

    def test_scale(self):
        h = get_hist(bh.axis.Regular(10, 10, 20), bh.axis.Integer(0, 10), flow=False)
        h2 = h.hist.scale(2.0, "var1")
        assert_allclose(h2.hist.edges("var1"), h.hist.edges("var1") * 2)
        assert_allclose(h2.hist.edges("var2"), h.hist.edges("var2"))


def test_normalization():
    x = get_array([50], name="var1")
    y = get_array([50], name="var2")
    z = get_array([50], name="var3")
    h = xh.histogramdd(x, y, z, range=[(0, 1)] * 3, density=False)
    h = h.hist.normalize()
    ref = xh.histogramdd(x, y, z, range=[(0, 1)] * 3, density=True)
    assert_allclose(h, ref)


class TestStatistics:
    ax = bh.axis.Regular(30, 0.0, 10.0)

    @property
    def tol(self) -> float:
        """We take tolerance as half the bins width."""
        return (self.ax.edges[-1] - self.ax.edges[0]) / self.ax.size / 2

    def get_values(self):
        rng = np.random.default_rng(seed=42)
        out = xr.DataArray(rng.normal(loc=5.0, size=100), dims=["x"], name="var")
        return out

    def get_hist(self, values: xr.DataArray) -> xr.DataArray:
        return xh.histogram(values, bins=self.ax)

    def test_median(self):
        values = self.get_values()
        hist = self.get_hist(values)
        assert np.isclose(hist.hist.median(), np.median(values), atol=self.tol)

    def test_mean(self):
        values = self.get_values()
        hist = self.get_hist(values)
        assert np.isclose(hist.hist.mean(), np.mean(values), atol=self.tol)

    def test_var(self):
        values = self.get_values()
        hist = self.get_hist(values)
        assert np.isclose(hist.hist.var(), np.var(values), atol=self.tol)
