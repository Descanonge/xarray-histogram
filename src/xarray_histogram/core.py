"""Main functions."""

# This file is part of the 'xarray-histogram' project
# (http://github.com/Descanonge/xarray-histogram) and subject
# to the MIT License as defined in the file 'LICENSE',
# at the root of this project. © 2022 Clément Haëck

from __future__ import annotations

import warnings
from collections import abc

import boost_histogram as bh
import numpy as np
import xarray as xr
from xarray.core.utils import is_dask_collection

try:
    import dask_histogram as dh
    from dask.array.core import Array as DaskArray

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


VAR_WEIGHT = "_weight"

AxisSpec = bh.axis.Axis | int | abc.Sequence[int | float]


class BinsMinMaxWarning(UserWarning):
    """Warning if range of bins is not supplied and must be computed from data."""

    pass


def histogram(
    *data: xr.DataArray,
    bins: abc.Sequence[AxisSpec],
    dims: abc.Collection[abc.Hashable] | None = None,
    weight: xr.DataArray | None = None,
    density: bool = False,
) -> xr.DataArray:
    """Compute histogram.

    Parameters
    ----------
    data
        The `xarray.DataArray` to compute the histogram from. To compute a
        multi-dimensional histogram supply a sequence of as many arrays
        as the histogram dimensionality. All arrays must have the same
        dimensions.
    bins
        Specification of the histograms bins. Supply a sequence of
        specifications for a multi-dimensional array, in the same order as
        the `data` arrays.

        Specification can either be:

        * a `boost-histogram.axis.Axis` or subtype.
        * a tuple of (number of bins, minimum value, maximum value) in which case the
          bins will be linearly spaced
        * the number of bins, the minimum and maximum values are computed from the data
          on the spot.
    dims
        Dimensions to compute the histogram along to. If left to None the
        data is flattened along all axis.
    weight
        Array of the weights to apply for each data-point. It will be broadcasted
        against the data arrays.
    density
        If true normalize the histogram so that its integral is one.
        Does not take into account `weight`. Default is false.


    Returns
    -------
    histogram
        DataArray named `hist_<variable name>` or just `hist` for
        multi-dimensional histograms. The bins coordinates are named
        `bins_<variable name>`.
    """
    in_data = list(data)
    data_sanity_check(in_data)
    variables = [a.name for a in in_data]

    bins = manage_bins_input(bins, in_data)
    bins_names = [f"{v}_bins" for v in variables]

    if weight is not None:
        weight = weight.rename(VAR_WEIGHT)
        in_data.append(weight)

    in_data = list(xr.broadcast(*in_data))

    if not is_all_dask(in_data):
        comp_hist_func = comp_hist_numpy
        for a in in_data:
            a.compute()
    else:
        comp_hist_func = comp_hist_dask

    # Merge everything together so it can be sent through a single
    # groupby call.
    ds = xr.merge(in_data, join="exact")
    data_dims = ds.dims

    if dims is None:
        dims = data_dims

    if set(dims) == set(data_dims):
        # on flattened array
        hist = comp_hist_func(ds, variables, bins, bins_names)
    else:
        # stack dimensions that we don't flatten
        stacked_dim = [d for d in data_dims if d not in dims]
        ds_stacked = ds.stack(stacked_dim=stacked_dim)

        hist = ds_stacked.groupby("stacked_dim").map(
            comp_hist_func, shortcut=True, args=[variables, bins, bins_names]
        )
        hist = hist.unstack()

    hist = hist.rename("hist")
    for name, b in zip(bins_names, bins, strict=True):
        hist = hist.assign_coords({name: b.edges[:-1]})
        hist[name].attrs["right_edge"] = b.edges[-1]

    if density:
        widths = [np.diff(b.edges) for b in bins]
        if len(widths) == 1:
            area = widths[0]
        elif len(widths) == 2:  # noqa: PLR2004
            area = np.outer(*widths)
        else:
            # https://stackoverflow.com/a/43149109
            # I added dtype=object to silence a warning for ragged arrays
            # not sure how safe this is.
            area = np.prod(np.array(np.ix_(*widths), dtype=object))

        area_xr = xr.DataArray(area, dims=bins_names)
        hist = hist / area_xr / hist.sum(bins_names)

    return hist


def separate_ravel(
    ds: xr.Dataset, variables: abc.Sequence[abc.Hashable]
) -> tuple[list[DaskArray] | list[np.ndarray], DaskArray | np.ndarray | None]:
    """Separate data and weight arrays and flatten arrays.

    Returns
    -------
    data
        List of data arrays.
    weight
        Array of weights if present in dataset, None otherwise.
    """
    data = [ds[v].data.ravel() for v in variables]
    if VAR_WEIGHT in ds:
        weight = ds[VAR_WEIGHT].data.ravel()
    else:
        weight = None
    return data, weight


def comp_hist_dask(
    ds: xr.Dataset,
    variables: abc.Sequence[abc.Hashable],
    bins: abc.Sequence[bh.axis.Axis],
    bins_names: abc.Sequence[abc.Hashable],
) -> xr.DataArray:
    """Compute histogram for dask data."""
    data, weight = separate_ravel(ds, variables)
    hist = dh.factory(*data, axes=bins, weights=weight)  # type: ignore
    res = hist.to_dask_array()

    return xr.DataArray(res[0], dims=bins_names)


def comp_hist_numpy(
    ds: xr.Dataset,
    variables: abc.Sequence[abc.Hashable],
    bins: abc.Sequence[bh.axis.Axis],
    bins_names: abc.Sequence[abc.Hashable],
) -> xr.DataArray:
    """Compute histogram for numpy data."""
    hist = bh.Histogram(*bins)
    data, weight = separate_ravel(ds, variables)
    hist.fill(*data, weight=weight)
    return xr.DataArray(hist.values(), dims=bins_names)


def data_sanity_check(data: abc.Sequence[xr.DataArray]):
    """Ensure data is correctly formated.

    Raises
    ------
    TypeError: If a 0-length sequence was supplied.
    TypeError: If any data is not a `xarray.DataArray`.
    ValueError: If set of dimensions are not identical in all arrays.
    """
    if len(data) == 0:
        raise TypeError("Data sequence of length 0.")
    for a in data:
        if not isinstance(a, xr.DataArray):
            raise TypeError(
                "Data must be a xr.DataArray, "
                f"a type {type(a).__name__} was supplied."
            )
    dims0 = set(data[0].dims)
    for a in data:
        if set(a.dims) != dims0:
            raise ValueError("Dimensions are different in supplied arrays.")


def weight_sanity_check(
    weight: abc.Sequence[xr.DataArray], data: abc.Sequence[xr.DataArray]
):
    """Ensure weight is correctly formated.

    Raises
    ------
    TypeError: If weight is not a `xarray.DataArray`.
    ValueError: If the set of dimensions are not the same in weights as data.
    """
    dims0 = set(data[0].dims)
    if not isinstance(weight, xr.DataArray):
        raise TypeError(
            "Weights must be a xr.DataArray, "
            f"a type {type(weight).__name__} was supplied."
        )
    if set(weight.dims) != dims0:
        raise ValueError("Dimensions are different in supplied weights.")
    # We only check for correct set of dimensions. Weight will be aligned
    # later with data anyway


def silent_minmax_warning():
    """Filter out the warning for computing min-max values."""
    warnings.filterwarnings("ignore", category=BinsMinMaxWarning)


def manage_bins_input(
    bins: abc.Sequence[AxisSpec], data: abc.Sequence[xr.DataArray]
) -> abc.Sequence[bh.axis.Axis]:
    """Check bins input and convert to boost objects.

    Raises
    ------
    ValueError: If there are not as much bins specifications as data arrays.
    """
    if len(bins) != len(data):
        raise ValueError(
            f"Not as much bins specifications ({len(bins)}) "
            f"as data arrays ({len(data)}) were supplied"
        )
    bins_out = []
    for spec, a in zip(bins, data):  # noqa: B905
        if isinstance(spec, bh.axis.Axis):
            bins_out.append(spec)
            continue
        if isinstance(spec, int):
            nbins = spec
            warnings.warn(
                (
                    "Range was not supplied, the minimum and maximum values "
                    "will have to be computed. use `silent_minmax_warning()` to "
                    "silent this warning."
                ),
                BinsMinMaxWarning,
                stacklevel=1,
            )
            start = float(a.min().values)
            stop = float(a.max().values)
        else:
            if len(spec) != 3:  # noqa: PLR2004
                raise IndexError(
                    f"Unexpected length of regular bins specification ({len(spec)})"
                )
            if not isinstance(spec[0], int):
                raise TypeError(
                    "First item of bins specification should be an integer: "
                    f"the number of bins. Received {type(spec[0])}"
                )

            nbins = spec[0]
            start, stop = spec[1:]
        bins_out.append(bh.axis.Regular(nbins, start, stop))

    return bins_out


def is_all_dask(data: abc.Sequence[xr.DataArray]) -> bool:
    """Check if all the variables are in dask format.

    Only return true if Dask and Dask-histogram are imported.
    """
    all_dask = HAS_DASK and all(is_dask_collection(a.data) for a in data)
    return all_dask
