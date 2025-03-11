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

    return h


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

    with pytest.raises(TypeError):
        h.hist._variable(None)

    with pytest.raises(KeyError):
        h.hist._variable("bad variable")


class TestEdges:
    def test_infer_right_edge(self) -> None:
        h = get_blank_histogram()
        # this reset right edge
        h = h.assign_coords(var1_bins=np.arange(0, 10))

        assert isinstance(h.hist, HistDataArrayAccessor)
        assert_allclose(h.hist.edges(), np.arange(0, 11))

        # Unregular edges
        h_wrong = h.assign_coords(var1_bins=np.logspace(1, 10, 10))
        with pytest.raises(ValueError):
            _ = h_wrong.hist

    def test_basic(self) -> None:
        h = get_blank_histogram()
        assert_allclose(h.hist.edges("var1"), np.arange(0, 11))

    def test_centers(self):
        h = get_blank_histogram()
        assert_allclose(h.hist.centers(), np.arange(0, 10) + 0.5)

    def test_width(self):
        h = get_blank_histogram()
        assert_allclose(h.hist.widths(), np.ones(10))

    def test_areas(self):
        # the accessor uses functools.reduce(operator.mul) over xr.DataArrays
        # we test against computation done with numpy
        h = get_blank_histogram(3)
        width1 = np.ones(10)

        # spice things up
        bins2 = np.linspace(0, 1, 11)  # width 0.1
        h = h.assign_coords(var2_bins=bins2[:-1])
        h.var2_bins.attrs["right_edge"] = bins2[-1]
        # width2 = np.full((10,), 0.1)
        width2 = np.diff(bins2)
        assert_allclose(h.hist.edges("var2"), bins2)
        assert_allclose(h.hist.widths("var2"), width2)

        bins3 = np.logspace(0.1, 1, 11)
        h = h.assign_coords(var3_bins=bins3[:-1])
        h.var3_bins.attrs["right_edge"] = bins3[-1]
        width3 = np.diff(bins3)
        assert_allclose(h.hist.edges("var3"), bins3)
        assert_allclose(h.hist.widths("var3"), width3)

        assert_allclose(h.hist.areas(["var1"]), width1)
        assert_allclose(h.hist.areas(["var1", "var2"]), np.outer(width1, width2))
        assert_allclose(h.hist.areas(["var1", "var3"]), np.outer(width1, width3))
        assert_allclose(
            h.hist.areas(), reduce(np.multiply, np.ix_(width1, width2, width3))
        )


class TestTransformBins:
    def test_apply(self):
        h = get_blank_histogram(2)
        h2 = h.hist.apply_func(lambda bins: bins + 2, "var1")
        assert_allclose(h2.hist.edges("var1"), h.hist.edges("var1") + 2)
        assert_allclose(h2.hist.edges("var2"), h.hist.edges("var2"))

    def test_scale(self):
        h = get_blank_histogram(2)
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
