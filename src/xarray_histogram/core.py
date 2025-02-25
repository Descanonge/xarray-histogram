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

try:
    from dask.base import is_dask_collection, tokenize
    from dask.blockwise import Blockwise, blockwise
    from dask.highlevelgraph import HighLevelGraph
    from dask_histogram.core import PartitionedHistogram, _dependencies

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


VAR_WEIGHTS = "_weights"
VAR_HIST = "__hist"
LOOP_DIM = "__loop_var"

AxisSpec = bh.axis.Axis | int | abc.Sequence[int | float]
"""Accepted input types for bins specification."""


class BinsMinMaxWarning(UserWarning):
    """Warning if range of bins is not supplied and must be computed from data."""


def histogram(
    *data: xr.DataArray,
    bins: abc.Sequence[AxisSpec],
    dims: abc.Collection[abc.Hashable] | None = None,
    weights: xr.DataArray | None = None,
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
    weights
        Array of the weights to apply for each data-point.
    density
        If true normalize the histogram so that its integral is one.
        Does not take into account `weight`. Default is false.
    kwargs
        Passed to :class:`boost_histogram.Histogram` initialization.

    Returns
    -------
    histogram
        DataArray named ``<variables names>_histogram`` (for multi-dimensional
        histograms the names are separated by underscores). The bins coordinates are
        named ``<variable name>_bins``.
    """
    data_sanity_check(data)
    variables = [a.name for a in data]

    axes = manage_bins_input(bins, data)
    bins_names = [f"{v}_bins" for v in variables]

    if weights is not None:
        weights = weights.rename(VAR_WEIGHTS)
        data = data + (weights,)

    data = xr.broadcast(*data)

    if is_any_dask(data):
        histogram_func = histogram_dask
        data = tuple(a.chunk({}) for a in data)
        data = xr.unify_chunks(*data)  # type: ignore[assignment]
    else:
        histogram_func = histogram_numpy

    # Merge everything together so it can be sent through a single
    # groupby call.
    ds = xr.merge(data, join="exact")

    data_dims = ds.dims
    if dims is None:
        dims = data_dims
    # dimensions that we loop over
    dims_loop = set(data_dims) - set(dims)

    if len(dims_loop) == 0:
        # on flattened array
        hist = histogram_func(ds, variables, axes, bins_names, **kwargs)[VAR_HIST]
    else:
        hist_ds = ds.groupby({d: xr.groupers.UniqueGrouper() for d in dims_loop}).map(
            histogram_func,
            shortcut=True,
            args=(variables, axes, bins_names),
            **kwargs,
        )
        hist = hist_ds[VAR_HIST]

    for name, b in zip(bins_names, axes, strict=True):
        hist = hist.assign_coords({name: b.edges[:-1]})
        hist[name].attrs["right_edge"] = b.edges[-1]

    if density:
        widths = [np.diff(b.edges) for b in axes]
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

    hist_name = "pdf" if density else "histogram"
    hist = hist.rename("_".join(map(str, variables + [hist_name])))
    return hist


def _clone(histref: bh.Histogram) -> bh.Histogram:
    return bh.Histogram(*histref.axes, storage=histref.storage_type())


def _blocked_dd(*data, histref: bh.Histogram) -> bh.Histogram:
    """Multiple variables, ND arrays, unweighted.

    Arrays are already broadcasted.
    """
    thehist = _clone(histref)
    flattened = (np.reshape(x, (-1,)) for x in data)
    return thehist.fill(*flattened)


def _blocked_dd_w(*data, histref: bh.Histogram) -> bh.Histogram:
    """Multiple variables, ND arrays, weighted.

    Arrays are already broadcasted.
    """
    thehist = _clone(histref)
    flattened = (np.reshape(x, (-1,)) for x in data)
    *data_flat, weights_flat = flattened
    return thehist.fill(*data_flat, weight=weights_flat)


def _partitionwise(func, layer_name, *args, **kwargs) -> Blockwise:
    pairs = []
    numblocks = {}
    for arg in args:
        numblocks[arg._name] = arg.numblocks
        pairs.extend([arg.name, "".join(chr(97 + i) for i in range(arg.ndim))])

    return blockwise(
        func,
        layer_name,
        "a",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )


def histogram_dask(
    ds: xr.Dataset,
    variables: abc.Sequence[abc.Hashable],
    axes: abc.Sequence[bh.axis.Axis],
    bins_names: abc.Sequence[abc.Hashable],
    **kwargs,
) -> xr.Dataset:
    """Compute histogram for dask data."""
    histref = bh.Histogram(*axes, **kwargs)

    data = [ds[v].data for v in variables]
    if VAR_WEIGHTS in ds:
        weights = ds[VAR_WEIGHTS].data
    else:
        weights = None

    name = f"hist-on-block-{tokenize(data, histref, weights)}"
    if weights is not None:
        g = _partitionwise(_blocked_dd_w, name, *data, weights, histref=histref)
    else:
        g = _partitionwise(_blocked_dd, name, *data, histref=histref)

    dependencies = _dependencies(*data, weights=weights)
    hlg = HighLevelGraph.from_collections(name, g, dependencies=dependencies)  # type: ignore[arg-type]

    npartitions = len(g.get_output_keys())
    hist = PartitionedHistogram(hlg, name, npartitions, histref=histref)

    values, *_ = hist.collapse().to_dask_array()
    return xr.DataArray(values, dims=bins_names, name=VAR_HIST).to_dataset()


def histogram_numpy(
    ds: xr.Dataset,
    variables: abc.Sequence[abc.Hashable],
    axes: abc.Sequence[bh.axis.Axis],
    bins_names: abc.Sequence[abc.Hashable],
    **kwargs,
) -> xr.Dataset:
    """Compute histogram for numpy data."""
    histref = bh.Histogram(*axes, **kwargs)

    data = [ds[v].data for v in variables]
    if VAR_WEIGHTS in ds:
        weights = ds[VAR_WEIGHTS].data
        hist = _blocked_dd_w(*data, weights, histref=histref)
    else:
        hist = _blocked_dd(*data, histref=histref)

    return xr.DataArray(hist.values(), dims=bins_names, name=VAR_HIST).to_dataset()


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
                f"Data must be a xr.DataArray, a type {type(a).__name__} was supplied."
            )


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
    if not isinstance(weight, xr.DataArray):
        raise TypeError(
            "Weights must be a xr.DataArray, "
            f"a type {type(weight).__name__} was supplied."
        )


def silent_minmax_warning():
    """Filter out the warning for computing min-max values."""
    warnings.filterwarnings("ignore", category=BinsMinMaxWarning)


def manage_bins_input(
    bins: abc.Sequence[AxisSpec], data: abc.Sequence[xr.DataArray]
) -> tuple[bh.axis.Axis, ...]:
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

    return tuple(bins_out)


def is_any_dask(data: abc.Sequence[xr.DataArray]) -> bool:
    """Check if any the variables are in dask format.

    Only return true if Dask and Dask-histogram are imported.
    """
    return HAS_DASK and any(is_dask_collection(a.data) for a in data)
