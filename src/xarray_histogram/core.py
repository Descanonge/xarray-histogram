"""Main functions."""

# This file is part of the 'xarray-histogram' project
# (http://github.com/Descanonge/xarray-histogram) and subject
# to the MIT License as defined in the file 'LICENSE',
# at the root of this project. © 2022 Clément Haëck

from __future__ import annotations

import operator
from collections import abc
from copy import copy
from functools import partial, reduce

import boost_histogram as bh
import numpy as np
import xarray as xr
from numpy.typing import NDArray

try:
    import dask.array as da
    from dask.base import is_dask_collection

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


# TODO Forbid growth when using dask

_range = range

VAR_WEIGHTS = "_weights"
VAR_HIST = "__hist"
LOOP_DIM = "__loop_var"

BinsType = bh.axis.Axis | int
RangeType = tuple[float | None, float | None]


def histogram(
    x: xr.DataArray,
    /,
    bins: BinsType = 10,
    range: RangeType | None = None,
    weights: xr.DataArray | None = None,
    density: bool = False,
    dims: abc.Collection[abc.Hashable] | None = None,
    **kwargs,
) -> xr.DataArray:
    return histogramdd(
        x,
        bins=bins,
        range=[range] if range is not None else None,
        weights=weights,
        density=density,
        dims=dims,
        **kwargs,
    )


def histogram2d(
    x: xr.DataArray,
    y: xr.DataArray,
    /,
    bins: BinsType | abc.Sequence[BinsType] = 10,
    range: abc.Sequence[RangeType] | None = None,
    weights: xr.DataArray | None = None,
    density: bool = False,
    dims: abc.Collection[abc.Hashable] | None = None,
    **kwargs,
) -> xr.DataArray:
    return histogramdd(
        x,
        y,
        bins=bins,
        range=range,
        weights=weights,
        density=density,
        dims=dims,
        **kwargs,
    )


def histogramdd(
    *data: xr.DataArray,
    bins: abc.Sequence[BinsType] | BinsType = 10,
    range: abc.Sequence[RangeType] | None = None,
    weights: xr.DataArray | None = None,
    density: bool = False,
    dims: abc.Collection[abc.Hashable] | None = None,
    storage: bh.storage.Storage | None = None,
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
    variables = [a.name for a in data]
    bins_names = [f"{v}_bins" for v in variables]
    axes = get_axes_from_specs(bins, range, data)

    if storage is None:
        storage = bh.storage.Double()
    histref = bh.Histogram(*axes, storage=storage)

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
        areas = xr.DataArray(reduce(operator.mul, widths), dims=bins_names)
        hist = hist / areas / hist.sum(bins_names)

    hist_name = "pdf" if density else "histogram"
    hist = hist.rename("_".join(map(str, variables + [hist_name])))
    return hist


def _clone(histref: bh.Histogram) -> bh.Histogram:
    return bh.Histogram(*histref.axes, storage=histref.storage_type())


def _blocked_dd(*data: NDArray, weight: bool, histref: bh.Histogram) -> NDArray:
    """Multiple variables, ND arrays.

    Arrays are already broadcasted.
    """
    thehist = _clone(histref)
    flattened = (np.reshape(x, (-1,)) for x in data)

    thehist = _clone(histref)
    if weight:
        *args, weights = flattened
        thehist.fill(*args, weight=weights)
    else:
        thehist.fill(*flattened)

    return thehist.values()


def _blocked_dd_loop(
    *data: NDArray,
    axis_agg: abc.Sequence[int],
    axis_loop: abc.Sequence[int],
    weight: bool,
    histref: bh.Histogram,
) -> NDArray:
    """Multiple variables, ND arrays.

    Arrays are already broadcasted.
    """
    # move aggregated axis at the end
    ndim = data[0].ndim
    ordered = [
        np.moveaxis(a, axis_agg, tuple(_range(ndim - len(axis_agg), ndim)))
        for a in data
    ]

    shape = data[0].shape
    shape_agg = tuple(shape[i] for i in axis_agg)
    n_agg = reduce(operator.mul, shape_agg)
    shape_loop = tuple(shape[i] for i in axis_loop)
    n_loop = reduce(operator.mul, shape_loop)
    flattened = [np.reshape(a, (n_loop, n_agg)) for a in ordered]

    values = np.zeros((n_loop, *histref.shape))
    for i in range(n_loop):
        thehist = _clone(histref)
        if weight:
            *args, weights = flattened
            thehist.fill(*[a[i] for a in args], weight=weights[i])
        else:
            thehist.fill(*[a[i] for a in flattened])

        # TODO: take care of flow
        values[i] = thehist.values()

    out = np.reshape(values, (*shape_loop, *histref.shape))
    out = da.expand_dims(
        out, tuple(_range(len(axis_loop), len(axis_loop) + len(axis_agg)))
    )

    return out


def _histogram_dask(
    *data: da.Array,
    axis_loop: abc.Sequence[int],
    axis_agg: abc.Sequence[int],
    weight: bool,
    histref: bh.Histogram,
) -> da.Array:
    """Compute histogram for dask data."""
    # TODO: Use functools.partial instead, _bocked_dd does not have the same signature
    func = _blocked_dd_loop if axis_loop else _blocked_dd
    dtype = (
        int
        if histref.storage_type in (bh.storage.Int64, bh.storage.AtomicInt64)
        else float
    )
    blocked = da.map_blocks(
        func,
        *data,
        axis_loop=axis_loop,
        axis_agg=axis_agg,
        weight=weight,
        histref=histref,
        dtype=dtype,
        chunks=(
            *[data[0].chunks[i][0] for i in axis_loop],
            *[1 for _ in axis_agg],
            *histref.shape,
        ),
        new_axis=[data[0].ndim + i for i in range(histref.ndim)],
        enforce_ndim=True,
        name="hist-on-block",
        meta=np.array((), dtype=dtype),
    )
    return blocked.sum(axis_agg)


def get_axes_from_specs(
    bins: abc.Sequence[BinsType] | BinsType,
    ranges: abc.Sequence[RangeType] | None,
    data: abc.Sequence[xr.DataArray],
) -> tuple[bh.axis.Axis, ...]:
    """Check bins input and convert to boost objects.

    Raises
    ------
    ValueError
        If there are not as much bins specifications as data arrays.
    """
    if isinstance(bins, bh.axis.Axis | int):
        bins = [copy(bins) for _ in _range(len(data))]
    if ranges is None:
        ranges = [(None, None) for _ in _range(len(data))]

    if len(bins) != len(data):
        raise IndexError(
            f"Not as much bins specifications ({len(bins)}) "
            f"as data arrays ({len(data)}) were supplied"
        )
    if len(ranges) != len(data):
        raise IndexError(
            f"Not as much range specifications ({len(ranges)}) "
            f"as data arrays ({len(data)}) were supplied"
        )

    axes = []
    for spec, range, a in zip(bins, ranges, data, strict=False):
        if isinstance(spec, bh.axis.Axis):
            axes.append(spec)
        elif isinstance(spec, int):
            start, stop = range
            if start is None:
                start = float(a.min())
            if stop is None:
                stop = float(a.max())

            axes.append(bh.axis.Regular(spec, start, stop))
        else:
            raise TypeError(
                "Bins must be specified as boost Axis or with an integer. "
                f"Received {type(spec)}."
            )

    # TODO check for growth

    return tuple(axes)


def is_any_dask(data: abc.Sequence[xr.DataArray]) -> bool:
    """Check if any the variables are in dask format.

    Only return true if Dask and Dask-histogram are imported.
    """
    return HAS_DASK and any(is_dask_collection(a.data) for a in data)
