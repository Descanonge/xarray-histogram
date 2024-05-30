"""Accessor to manipulate histograms.

An accessor registered as ``hist`` is made available on :class:`xarray.DataArray` for
various histogram manipulations.
"""

import re
import typing as t
from collections import abc

import numpy as np
import xarray as xr
from scipy.stats import rv_histogram


class HistAccessor:
    """Common Dataset and DataArray accessor functionality."""

    pass


@xr.register_dataarray_accessor("hist")
class HistDataArrayAccesor(HistAccessor):
    """Histogram accessor for DataArrays.

    .. important::

        Accessor registered under ``hist``.

    .. rubric:: Validity

    They are some conditions for the accessor to be accessible:

    * The array must be named as ``<variable(s)_name>_<histogram or pdf>``. `histogram`
      if it is *not* normalized, and `pdf` if it is normalized as a probability density
      function. If the histogram is multi-dimensional, the variables names must be
      separated by underscores. For instance: ``Temp_Sal_histogram``.
    * The variables found in the array name will be used to find the bins dimensions,
      which must be named ``<variable>_bins``.
    * Each bins coordinates must contain an attribute named `right_edge`, corresponding
      to the right edge of the last bin.

    Those conventions are coherent with the output of :func:`.core.histogram`, so if you
    use this function you should not have to worry.

    .. rubric:: Backend for computations

    Most computations are actually delegated to :class:`scipy.stats.rv_histogram`.
    Therefore, it does not support chunking along the bins dimensions (which should not
    be a problem in most case).
    """

    _NAME_PATTERN = re.compile(".+?_(?:.+?_)+(histogram|pdf)")
    _VALID_NAMES: list[str] = ["histogram", "pdf"]

    def __init__(self, obj: xr.DataArray) -> None:
        self._obj = obj
        self._variable_type: str | None = None
        self._variables: list[str] | None = None

        self._check_validity()

    @property
    def variable_type(self) -> str:
        """Kind of histogram (one of "histogram" or "pdf")."""
        if self._variable_type is not None:
            return self._variable_type
        m = self._NAME_PATTERN.fullmatch(str(self._obj.name))
        if m is None:
            raise ValueError(f"Invalid array name '{self._obj.name}'")
        name = m.group(1)
        if name not in self._VALID_NAMES:
            raise KeyError(
                f"Invalid variable-type name '{name}'. "
                f"Must be one of {self._VALID_NAMES}"
            )
        self._variable_type = name
        return name

    @property
    def density(self) -> bool:
        """Whether the histogram is normalized (based on the array name)."""
        return self._variable_type == "pdf"

    @property
    def variables(self) -> list[str]:
        """List of variables included in the histograms."""
        if self._variables is not None:
            return self._variables
        name = str(self._obj.name)
        variables = name.removesuffix("_" + self.variable_type).split("_")
        if not variables:
            raise ValueError(f"No variables found in DataArray name '{name}'")
        self._variables = variables
        return self._variables

    def _dim(self, variable: str) -> str:
        """Transform a variable name to its corresponding dimension."""
        return f"{variable}_bins"

    def _check_validity(self) -> None:
        """Check validity of histogram.

        Check the variables names have corresponding bins dimensions, and if the
        coordinates contain a `right_edge` attribute.
        """
        obj = self._obj
        for var in self.variables:
            bin_dim = self._dim(var)
            if bin_dim not in obj.dims or bin_dim not in obj.coords:
                raise KeyError(f"No bin coordinates '{bin_dim}'")
            c = obj.coords[bin_dim]
            if "right_edge" not in c.attrs:
                raise KeyError(f"No attribute 'right edge' in '{bin_dim}'.")

    def edges(self, variable: str | None = None) -> xr.DataArray:
        """Return the edges of the bins (including the right most edge)."""
        if variable is None:
            variable = self.variables[0]
        dim = self._dim(variable)
        coord = self._obj.coords[dim]
        re = coord.attrs["right_edge"]
        re_xr = xr.DataArray(re, dims=[dim], coords={dim: [re]})
        full = xr.concat([coord, re_xr], dim=dim)
        return full

    def centers(self, variable: str | None = None) -> xr.DataArray:
        """Return the center of all bins."""
        edges = self.edges(variable)
        return (edges[:-1] + edges[1:]) / 2.0

    def widths(self, variable: str | None = None) -> xr.DataArray:
        """Return the width of all bins."""
        if variable is None:
            variable = self.variables[0]
        return self.edges(variable).diff(self._dim(variable))

    def normalize(
        self, variables: str | abc.Sequence[str] | None = None
    ) -> xr.DataArray:
        """Return a normalized histogram.

        Will raise if the histogram is already normalized.

        Parameters
        ----------
        variables
            The variable(s), ie dimensions, along which to normalize.
        """
        hist = self._obj
        if self.variable_type == "histogram":
            raise TypeError(f"'{hist.name}' already normalized.")

        if variables is None:
            variables = self.variables
        elif isinstance(variables, str):
            variables = [variables]

        widths = [self.widths(var) for var in variables]
        dims = [self._dim(var) for var in variables]
        if len(variables) == 1:
            area = widths[0]
        elif len(variables) == 2:  # noqa: PLR2004
            area = np.outer(*widths)  # type: ignore
        else:
            area = np.prod(np.ix_(*widths))

        area_xr = xr.DataArray(area, dims=dims)
        output = hist / area_xr / hist.sum(dims)
        output = output.rename("_".join(variables) + "_pdf")
        return output

    def apply_func(
        self,
        func: abc.Callable[[xr.DataArray], xr.DataArray],
        variable: str | None = None,
        **kwargs,
    ) -> xr.DataArray:
        """Apply a function to a bins coordinate.

        Parameters
        ----------
        func
            Callable that must transform the N+1 edges. It does not need to take care
            of the `right_edge` attribute.
        variable
            The variable to transform. (This is equivalent to computing an histogram of
            ``func(ds["variable"], **kwargs)``).
        kwargs
            Passed to the function.
        """
        if variable is None:
            variable = self.variables[0]
        dim = self._dim(variable)
        edges = self.edges(variable)
        new_edges = func(edges, **kwargs)
        new_edges.attrs["right_edge"] = new_edges[-1]
        return self._obj.assign_coords({dim: new_edges[:-1]})

    def scale(self, factor: float, variable: str | None = None) -> xr.DataArray:
        """Transform a bins coordinate by scaling it.

        Parameters
        ----------
        factor
            Factor by which to scale the coordinate values.
        variable
            The variable to scale. (This is equivalent to computing an histogram of
            ``factor * ds["variable"]``).
        """
        return self.apply_func(lambda arr: arr * factor, variable)

    def _apply_rv_func(
        self, func: str, variable: str | None = None, **kwargs
    ) -> xr.DataArray:
        """Apply a method of :class:`~scipy.stats.rv_histogram`.

        Parameters
        ----------
        func
            Name of the :class:`scipy.stats.rv_histogram` method to apply.
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        kwargs
            Passed to the function.
        """
        if variable is None:
            variable = self.variables[0]

        dim = self._dim(variable)
        bins = self.edges(variable)
        density = self.density

        def wrapped(arr: np.ndarray) -> float:
            if np.all(np.isnan(arr)):
                return np.nan
            rv_hist = rv_histogram((arr, bins), density=density)
            return getattr(rv_hist, func)(**kwargs)

        output = xr.apply_ufunc(
            wrapped,
            self._obj,
            input_core_dims=[[dim]],
            output_core_dims=[[]],
            output_dtypes=[float],
            dask="parallelized",
            vectorize=True,
        )

        output = output.rename(f"{variable}_{func}")
        return output

    def ppf(self, q: float, variable: str | None = None) -> xr.DataArray:
        """Return the percent point function at `q`.

        Parameters
        ----------
        q
            Must be between 0 and 1.
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        if not (0 < q < 1):
            raise ValueError(f"`q` must be between 0 and 1. (received {q})")
        return self._apply_rv_func("ppf", variable, q=q)

    def median(self, variable: str | None = None) -> xr.DataArray:
        """Return the median value of the distribution.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("median", variable)

    def mean(self, variable: str | None = None) -> xr.DataArray:
        """Return the mean value of the distribution.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("mean", variable)

    def cdf(self, x: float, variable: str | None = None) -> xr.DataArray:
        """Return the cumulative distribution function at `x`.

        Parameters
        ----------
        x
            Quantile, must be between 0 and 1.
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("cdf", variable)

    def var(self, variable: str | None = None) -> xr.DataArray:
        """Return the variance of the distribution.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("var", variable)

    def std(self, variable: str | None = None) -> xr.DataArray:
        """Return the standard deviation of the distribution.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("std", variable)

    def moment(self, order: int, variable: str | None = None) -> xr.DataArray:
        """Return the nth moment of the distribution.

        Parameters
        ----------
        order
            Order of moment, ``order>=1``.
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("moment", variable, order=order)

    def interval(self, confidence: float, variable: str | None = None) -> xr.Dataset:
        """Return the confidence interval with equal areas around the median.

        The interval is computed as ``[ppf(p_tail); ppf(1-p_tail)]`` with
        ``p_tail = (1-confidence)/2``.

        Parameters
        ----------
        confidence
            Probability that a value falls within the returned range. Must be between
            0 and 1.
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.

        Returns
        -------
        dataset
            Dataset with variables `confidence_low` and `confidence_high`, corresponding
            to the low and high values of the confidence interval.
        """
        if not (0 < confidence < 1):
            raise ValueError(
                f"Confidence must be between 0 and 1. (received {confidence})"
            )
        p_tail = (1 - confidence) / 2
        low = self._apply_rv_func("ppf", q=p_tail)
        high = self._apply_rv_func("ppf", q=1 - p_tail)
        output = xr.Dataset(dict(confidence_low=low, confidence_high=high))
        return output
