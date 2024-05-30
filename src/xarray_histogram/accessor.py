import re
import typing as t
from collections import abc

import numpy as np
import xarray as xr
from scipy.stats import rv_histogram


class HistAccessor:
    """Common Dataset and DataArray accessor functionality."""

    pass


class ApplicableFunction(t.Protocol):
    @t.overload
    def __call__(self, __x: xr.DataArray) -> xr.DataArray: ...

    @t.overload
    def __call__(self, __x: float) -> float: ...

    def __call__(self, __x: xr.DataArray | float) -> xr.DataArray | float: ...


@xr.register_dataarray_accessor("hist")
class HistDataArrayAccesor(HistAccessor):
    _NAME_PATTERN = re.compile(".+?_(?:.+?_)+(histogram|pdf)")
    _VALID_NAMES: list[str] = ["histogram", "pdf"]

    def __init__(self, obj: xr.DataArray) -> None:
        self._obj = obj
        self._variable_type: str | None = None
        self._variables: list[str] | None = None

        self._check_validity()

    @property
    def variable_type(self) -> str:
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
        return self._variable_type == "pdf"

    @property
    def variables(self) -> list[str]:
        if self._variables is not None:
            return self._variables
        name = str(self._obj.name)
        variables = name.removesuffix("_" + self.variable_type).split("_")
        if not variables:
            raise ValueError(f"No variables found in DataArray name '{name}'")
        self._variables = variables
        return self._variables

    def _dim(self, variable: str) -> str:
        return f"{variable}_bins"

    def _check_validity(self) -> None:
        obj = self._obj
        for var in self.variables:
            bin_dim = self._dim(var)
            if bin_dim not in obj.dims or bin_dim not in obj.coords:
                raise KeyError(f"No bin coordinates '{bin_dim}'")
            c = obj.coords[bin_dim]
            if "right_edge" not in c.attrs:
                raise KeyError(f"No attribute 'right edge' in '{bin_dim}'.")

    def edges(self, variable: str | None = None) -> xr.DataArray:
        if variable is None:
            variable = self.variables[0]
        dim = self._dim(variable)
        coord = self._obj.coords[dim]
        re = coord.attrs["right_edge"]
        re_xr = xr.DataArray(re, dims=[dim], coords={dim: [re]})
        full = xr.concat([coord, re_xr], dim=dim)
        return full

    def centers(self, variable: str | None = None) -> xr.DataArray:
        edges = self.edges(variable)
        return (edges[:-1] + edges[1:]) / 2.0

    def widths(self, variable: str | None = None) -> xr.DataArray:
        if variable is None:
            variable = self.variables[0]
        return self.edges(variable).diff(self._dim(variable))

    def normalize(
        self, variables: str | abc.Sequence[str] | None = None
    ) -> xr.DataArray:
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
        func: ApplicableFunction,
        variable: str | None = None,
        **kwargs,
    ) -> xr.DataArray:
        if variable is None:
            variable = self.variables[0]
        dim = self._dim(variable)
        coord = self._obj.coords[dim]
        right_edge = func(coord.attrs["right_edge"], **kwargs)
        new_coord = func(coord, **kwargs)
        new_coord.attrs["right_edge"] = right_edge

        return self._obj.assign_coords({dim: new_coord})

    def scale(self, factor: float, variable: str | None = None) -> xr.DataArray:
        return self.apply_func(lambda arr: arr * factor, variable)

    def _apply_rv_func(
        self, func: str, variable: str | None = None, **kwargs
    ) -> xr.DataArray:
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
        if not (0 < q < 1):
            raise ValueError(f"`q` must be between 0 and 1. (received {q})")
        return self._apply_rv_func("ppf", variable, q=q)

    def median(self, variable: str | None = None) -> xr.DataArray:
        return self._apply_rv_func("median", variable)

    def mean(self, variable: str | None = None) -> xr.DataArray:
        return self._apply_rv_func("mean", variable)

    def cdf(self, variable: str | None = None) -> xr.DataArray:
        return self._apply_rv_func("cdf", variable)

    def var(self, variable: str | None = None) -> xr.DataArray:
        return self._apply_rv_func("var", variable)

    def std(self, variable: str | None = None) -> xr.DataArray:
        return self._apply_rv_func("std", variable)

    def moment(self, order: int, variable: str | None = None) -> xr.DataArray:
        return self._apply_rv_func("moment", variable, order=order)

    def interval(self, confidence: float, variable: str | None = None) -> xr.Dataset:
        if not (0 < confidence < 1):
            raise ValueError(
                f"Confidence must be between 0 and 1. (received {confidence})"
            )
        p_tail = (1 - confidence) / 2
        low = self._apply_rv_func("ppf", q=p_tail)
        high = self._apply_rv_func("ppf", q=1 - p_tail)
        output = xr.Dataset(dict(confidence_low=low, confidence_high=high))
        return output
