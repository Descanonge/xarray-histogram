"""Accessor to manipulate histograms.

An accessor registered as ``hist`` is made available on :class:`xarray.DataArray` for
various histogram manipulations.
"""

from collections import abc

import numpy as np
import xarray as xr
from scipy.stats import rv_histogram

from .core import _bins_name, get_area, get_edges, get_widths, normalize


@xr.register_dataarray_accessor("hist")
class HistDataArrayAccessor:
    """Histogram accessor for DataArrays.

    .. important::

        Accessor registered under ``hist``.

    .. rubric:: Validity

    * The coordinates of the bins must be named ``<variable>_bins``.
    * The array must be named as ``<variable(s)_name>_<histogram or pdf>``. histogram*
    *if it is not normalized, and *pdf* if it is normalized as a probability density
    *function. If the histogram is multi-dimensional, the variables names must be
    *separated by underscores. For instance: ``Temp_Sal_histogram``.

    Each bins coordinate may contain attributes:

    * ``bin_type``: the class name of the Boost axis type that was used. If not present,
    the accessor will assume the bins are regularly spaced and will try to infer the
    rightmost edge. * ``right_edge``: the rightmost edge position, only necessary for
    Regular and Variable bins. * ``underflow`` and ``overflow``: booleans that indicate
    if the corresponding flow bins are present. If not present, will assume no flow
    bins.

    .. rubric:: Backend for computations

    Statistics computations are actually delegated to :class:`scipy.stats.rv_histogram`.
    Therefore, it does not support chunking along the bins dimensions (which should not
    be a problem in most cases).
    """

    _VALID_TYPES: list[str] = ["histogram", "pdf"]
    _variable_type: str
    variables: list[str]

    def __init__(self, obj: xr.DataArray) -> None:
        self._obj = obj

        # Fetch variables
        self.variables = []
        for dim in self._obj.dims:
            if str(dim).endswith("_bins"):
                self.variables.append(str(dim).removesuffix("_bins"))
        if not self.variables:
            raise ValueError(
                f"No bins coordinates found in DataArray '{self._obj.name}'"
            )
        self._check_name()
        self._check_bins()

        self._variable_type = str(self._obj.name).split("_")[-1]

    def __repr__(self) -> str:
        return f"Histogram accessor for {self._obj.name}"

    def _check_name(self) -> None:
        """Return if DataArray name is valid."""
        name = str(self._obj.name)
        err = f"Malformed array name '{name}'. "

        self._variable_type = ""
        for hist_type in self._VALID_TYPES:
            if name.endswith(f"_{hist_type}"):
                self._variable_type = hist_type
                break
        if not self._variable_type:
            raise ValueError(err + f"Name should end in one of {self._VALID_TYPES}")

        # Check all variables are accounted for
        unaccounted = list(self.variables)
        for _ in range(len(self.variables)):
            for var in unaccounted:
                found = False
                if name.startswith(var):
                    found = True
                    name = name.removeprefix(var + "_")
                    unaccounted.remove(var)
                    break
            if not found:
                raise ValueError(
                    err
                    + "Name contain unrecognized variables "
                    + f"(found variables are {self.variables})."
                )
        if unaccounted:
            raise ValueError(
                err + f"Variables {unaccounted} are not present in DataArrayname."
            )

    def _check_bins(self) -> None:
        """Check validity of bins.

        Check the variables names have corresponding bins dimensions, and infer right
        edge if necessary.
        """
        obj = self._obj
        for var in self.variables:
            bin_dim = _bins_name(var)
            if bin_dim not in obj.dims or bin_dim not in obj.coords:
                raise KeyError(f"No bin coordinates '{bin_dim}'")
            c = obj.coords[bin_dim]
            # Default is Regular
            if "bin_type" not in c.attrs:
                c.attrs["bin_type"] = "Regular"
            # rightmost edge inference
            if c.attrs["bin_type"] == "Regular" and "right_edge" not in c.attrs:
                diff = np.diff(c)
                if not np.allclose(diff, diff[0]):
                    raise ValueError(
                        f"Cannot infer right edge: bins for {var} "
                        "are not regularly spaced."
                    )
                c.attrs["right_edge"] = (c[-1] + diff[0]).values.item()

    def is_normalized(self) -> bool:
        """Whether the histogram is normalized (based on the array name)."""
        return self._variable_type == "pdf"

    def _variable(self, var: str | None) -> str:
        """Sanitize variable argument."""
        if var is None:
            if len(self.variables) == 1:
                return self.variables[0]
            raise ValueError("A variable must be given.")
        if var not in self.variables:
            raise KeyError(
                f"'{var}' is not among the histogram variables ({self.variables})"
            )
        return var

    def bins(self, variable: str | None = None, flow: bool = True) -> xr.DataArray:
        """Return bins coordinates for a given variable.

        Parameters
        ----------
        variable
            Can be omitted for 1D histograms.
        flow
            Remove flow bins if False.
        """
        variable = self._variable(variable)
        dim = _bins_name(variable)
        coord = self._obj.coords[dim]
        if not flow:
            coord = remove_flow(coord)
        return coord

    def edges(self, variable: str | None = None, flow: bool = True) -> xr.DataArray:
        """Return the edges of the bins (including the right most edge).

        Not supported for bins of the discrete types "IntCategory" and "StrCategory".

        Parameters
        ----------
        variable
            Can be omitted for 1D histograms.
        flow
            Remove flow bins if False.
        """
        edges = get_edges(self.bins(variable))
        if not flow:
            edges = remove_flow(edges)
        return edges

    def centers(self, variable: str | None = None, flow: bool = True) -> xr.DataArray:
        """Return the center of all bins.

        Not supported for bin type "StrCategory". IntCategory bins centers are
        ``bins+0.5``. The centers of flow bins are the same as their position
        (``np.inf`` for instance).

        Parameters
        ----------
        variable
            Can be omitted for 1D histograms.
        flow
            Remove flow bins if False.
        """
        variable = self._variable(variable)
        dim = _bins_name(variable)

        coord = self.bins(variable)
        bin_type = coord.attrs["bin_type"]
        if bin_type == "StrCategory":
            raise TypeError(f"Centers not supported for bin type {bin_type}")
        if bin_type in ["Integer", "IntCategory"]:
            out = coord + 0.5
            if not flow:
                out = remove_flow(out)
            return out

        def center(coord):
            return coord.rolling({dim: 2}, center=True).sum().dropna(dim) / 2.0

        return self._apply_to_coord(center, variable, keep_right_edge=True, flow=flow)

    def widths(self, variable: str | None = None, flow: bool = True) -> xr.DataArray:
        """Return the widths of all bins.

        Widths of flow bins and StrCategory are 1.

        Parameters
        ----------
        variable
            Can be omitted for 1D histograms.
        flow
            Remove flow bins if False.
        """
        widths = get_widths(self.bins(variable))
        if not flow:
            widths = remove_flow(widths)
        return widths

    def areas(
        self, variables: abc.Sequence[str] | None = None, flow: bool = True
    ) -> xr.DataArray:
        """Return the areas of the bins.

        The product of the widths of all specified bins.
        The areas of points that correspond to a flow bin in at least one dimension is
        equal to one.

        Parameters
        ----------
        variables
            Variables to include the corresponding bins. If left to None, all variables
            are used.
        flow
            Remove flow bins if False.
        """
        if variables is None:
            variables = self.variables
        return get_area(*[self.bins(v, flow=flow) for v in variables])

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
        if self._variable_type == "pdf":
            raise TypeError(f"'{self._obj.name}' already normalized.")

        if variables is None:
            variables = self.variables
        elif isinstance(variables, str):
            variables = [variables]

        bins_all = [_bins_name(var) for var in self.variables]
        bins_normalize = [_bins_name(var) for var in variables]
        output = normalize(self._obj, bins_all, bins_normalize)
        output = output.rename("_".join(self.variables) + "_pdf")
        return output

    def remove_flow(self, variables: abc.Sequence[str] | None = None) -> xr.DataArray:
        """Remove flow bins.

        Parameters
        ----------
        variables
            Variables for which to remove flow bins. If not specified, remove for all
            variables.
        """
        if variables is None:
            variables = self.variables

        slices = {}
        for var in variables:
            coord = self.bins(var)
            underflow = coord.attrs.get("underflow", False)
            overflow = coord.attrs.get("overflow", False)
            slices[_bins_name(var)] = slice(
                1 if underflow else 0, -1 if overflow else None
            )

        out = self._obj.isel(slices)

        for var in variables:
            coord = out.coords[_bins_name(var)]
            coord.attrs["underflow"] = False
            coord.attrs["overflow"] = False
        return out

    def _apply_to_coord(
        self,
        func: abc.Callable[[xr.DataArray], xr.DataArray],
        variable: str,
        keep_right_edge: bool = False,
        flow: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        dim = _bins_name(variable)
        coord = self._obj.coords[dim]

        # save flow bins coordinates
        underflow = coord.attrs.get("underflow", False)
        overflow = coord.attrs.get("overflow", False)
        underflow_bin = coord[0]
        overflow_bin = coord[-1]

        try:
            edges = self.edges(variable)
            edges_is_coord = False
        except TypeError:
            edges = coord
            edges_is_coord = True

        slc = slice(1 if underflow else 0, -1 if overflow else None)
        new_edges = func(edges[slc], **kwargs)
        if dim not in new_edges.coords:
            new_edges = new_edges.assign_coords({dim: new_edges})

        if keep_right_edge or edges_is_coord:
            new_coord = new_edges
        else:
            new_coord = new_edges[:-1]

        if underflow:
            new_coord = xr.concat([underflow_bin, new_coord], dim=dim)
        if overflow:
            new_coord = xr.concat([new_coord, overflow_bin], dim=dim)

        new_coord.attrs.update(coord.attrs)
        if "right_edge" in coord.attrs:
            new_coord.attrs["right_edge"] = (new_edges[-1]).values.item()

        if not flow:
            new_coord = remove_flow(new_coord)

        return new_coord

    def apply_func(
        self,
        func: abc.Callable[[xr.DataArray], xr.DataArray],
        variable: str | None = None,
        flow: bool = True,
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
            ``func(ds["variable"], **kwargs)``). It can be omitted for 1D histograms.
        kwargs
            Passed to the function.
        """
        variable = self._variable(variable)
        dim = _bins_name(variable)
        new_coord = self._apply_to_coord(
            func, variable, keep_right_edge=False, **kwargs
        )
        return self._obj.assign_coords({dim: new_coord})

    def scale(
        self, factor: float, variable: str | None = None, flow: bool = True
    ) -> xr.DataArray:
        """Transform a bins coordinate by scaling it.

        Parameters
        ----------
        factor
            Factor by which to scale the coordinate values.
        variable
            The variable to scale. (This is equivalent to computing an histogram of
            ``factor * ds["variable"]``).
        """
        return self.apply_func(lambda arr: arr * factor, variable, flow=flow)

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
        variable = self._variable(variable)
        dim = _bins_name(variable)
        coord = self.bins(variable)
        if coord.attrs.get("underflow", False) or coord.attrs.get("overflow", False):
            no_flow = self.remove_flow([dim])
            return no_flow.hist._apply_rv_func(func, variable, **kwargs)

        def wrapped(arr: np.ndarray, edges: np.ndarray, density: bool) -> float:
            if np.all(np.isnan(arr)):
                return np.nan
            rv_hist = rv_histogram((arr, edges), density=density)
            return getattr(rv_hist, func)(**kwargs)

        edges = self.edges(variable)
        density = self.is_normalized()

        output = xr.apply_ufunc(
            wrapped,
            self._obj,
            kwargs=dict(edges=edges.to_numpy(), density=density),
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

        Uses :class:`scipy.stats.rv_histogram` for computation.

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

        Uses :class:`scipy.stats.rv_histogram` for computation.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("median", variable)

    def mean(self, variable: str | None = None) -> xr.DataArray:
        """Return the mean value of the distribution.

        Uses :class:`scipy.stats.rv_histogram` for computation.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("mean", variable)

    def cdf(self, x: float, variable: str | None = None) -> xr.DataArray:
        """Return the cumulative distribution function at `x`.

        Uses :class:`scipy.stats.rv_histogram` for computation.

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

        Uses :class:`scipy.stats.rv_histogram` for computation.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("var", variable)

    def std(self, variable: str | None = None) -> xr.DataArray:
        """Return the standard deviation of the distribution.

        Uses :class:`scipy.stats.rv_histogram` for computation.

        Parameters
        ----------
        variable
            Variable along which to apply this function. All `rv_histogram` functions
            apply to a 1D histogram, so we loop over all other dimensions.
        """
        return self._apply_rv_func("std", variable)

    def moment(self, order: int, variable: str | None = None) -> xr.DataArray:
        """Return the nth moment of the distribution.

        Uses :class:`scipy.stats.rv_histogram` for computation.

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

        Uses :class:`scipy.stats.rv_histogram` for computation.

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
        low = self._apply_rv_func("ppf", variable, q=p_tail)
        high = self._apply_rv_func("ppf", variable, q=1 - p_tail)
        output = xr.Dataset(dict(confidence_low=low, confidence_high=high))
        return output


def remove_flow(x: xr.DataArray) -> xr.DataArray:
    overflow = x.attrs.get("overflow", False)
    underflow = x.attrs.get("underflow", False)
    out = x[slice(1 if underflow else 0, -1 if overflow else None)]
    out.attrs["underflow"] = False
    out.attrs["overflow"] = True
    return out
