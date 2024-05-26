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
LOOP_DIM = "__loop_var"

AxisSpec = bh.axis.Axis | int | abc.Sequence[int | float]


class BinsMinMaxWarning(UserWarning):
    """Warning if range of bins is not supplied and must be computed from data."""


def histogram(
    *data: xr.DataArray,
    bins: abc.Sequence[AxisSpec],
    dims: abc.Collection[abc.Hashable] | None = None,
    weight: xr.DataArray | None = None,
    density: bool = False,
    **kwargs,
) -> xr.DataArray:
    """Compute histogram.

    Parameters
    ----------
    data
        The arrays to compute the histogram from. To compute a multi-dimensional
        histogram supply a sequence of as many arrays as the histogram dimensionality.
        Arrays must be broadcastable against each other. If any underlying data is a
        dask array, other inputs will be transformed into a dask array of a single
        chunk.
    bins
        Sequence of specifications for the histogram bins, in the same order as the
        variables of `data`.

        Specification can either be:

        * a :class:`boost_histogram.axis.Axis`.
        * a tuple of (number of bins, minimum value, maximum value) in which case the
          bins will be linearly spaced
        * only the number of bins, the minimum and maximum values are then computed from
          the data on the spot.
    dims
        Dimensions to compute the histogram along to. If left to None the
        data is flattened along all axis.
    weight
        Array of the weights to apply for each data-point.
    density
        If true normalize the histogram so that its integral is one.
        Does not take into account `weight`. Default is false.
    kwargs
        Passed to :func:`bh.Histogram` initialization or :func:`dh.partitioned_factory`
        (for dask input).

    Returns
    -------
    histogram
        DataArray named ``<variables names>_histogram`` (for multi-dimensional
        histograms the names are separated by underscores). The bins coordinates are
        named ``<variable name>_bins``.
    """
    data_sanity_check(data)
    variables = [a.name for a in data]

    bins = manage_bins_input(bins, data)
    bins_names = [f"{v}_bins" for v in variables]

    if weight is not None:
        weight = weight.rename(VAR_WEIGHT)
        data = data + (weight,)

    data = xr.broadcast(*data)

    if is_any_dask(data):
        comp_hist_func = comp_hist_dask
        data = tuple(a.chunk({}) for a in data)
        data = xr.unify_chunks(*data)  # type: ignore[assignment]
    else:
        comp_hist_func = comp_hist_numpy

    # Merge everything together so it can be sent through a single
    # groupby call.
    ds = xr.merge(data, join="exact")

    data_dims = ds.dims

    if dims is None:
        dims = data_dims

    # dimensions that we loop over
    dims_loop = set(data_dims) - set(dims)

    # dimensions that are chunked. We need to manually aggregate them
    dims_aggr: set[abc.Hashable] = set()
    for var in variables:
        for dim, sizes in ds[var].chunksizes.items():
            if any(s != ds.sizes[dim] for s in sizes):
                dims_aggr.add(dim)
    dims_aggr -= dims_loop

    if len(dims_loop) == 0:
        # on flattened array
        hist = comp_hist_func(ds, variables, bins, bins_names)
    else:
        stacked = ds.stack(__stack_loop=dims_loop)

        hist = stacked.groupby(LOOP_DIM, squeeze=False).map(
            comp_hist_func, shortcut=True, args=[variables, bins, bins_names]
        )
        hist = hist.unstack()

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

    hist = hist.rename("_".join(map(str, variables + ["histogram"])))
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
    hist = dh.partitioned_factory(*data, axes=bins, weights=weight)  # type: ignore
    values, *_ = hist.collapse().to_dask_array()
    return xr.DataArray(values, dims=bins_names)


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
    TypeError
        If a 0-length sequence was supplied.
    TypeError
        If any data is not a :class:`xarray.DataArray`.
    ValueError
        If set of dimensions are not identical in all arrays.
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
    TypeError
        If weight is not a :class:`xarray.DataArray`.
    ValueError
        If the set of dimensions are not the same in weights as data.
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
    ValueError
        If there are not as much bins specifications as data arrays.
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


def is_any_dask(data: abc.Sequence[xr.DataArray]) -> bool:
    """Check if any the variables are in dask format.

    Only return true if Dask and Dask-histogram are imported.
    """
    any_dask = HAS_DASK and any(is_dask_collection(a.data) for a in data)
    return any_dask
