"""Main functions."""

# This file is part of the 'xarray-histogram' project
# (http://github.com/Descanonge/xarray-histogram) and subject
# to the MIT License as defined in the file 'LICENSE',
# at the root of this project. © 2022 Clément Haëck

from __future__ import annotations

import operator
import warnings
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

SIMPLE_STORAGE: list[type[bh.storage.Storage]] = [
    bh.storage.Double,
    bh.storage.Unlimited,
    bh.storage.Int64,
    bh.storage.AtomicInt64,
]

_range = range

VAR_WEIGHTS = "_weights"
VAR_HIST = "__hist"
LOOP_DIM = "__loop_var"

BinsType = bh.axis.Axis | int
RangeType = tuple[float | None, float | None]


# TODO add flow


def histogram(
    x: xr.DataArray,
    /,
    bins: BinsType = 10,
    range: RangeType | None = None,
    weights: xr.DataArray | None = None,
    density: bool = False,
    dims: abc.Collection[abc.Hashable] | None = None,
    storage: bh.storage.Storage | None = None,
) -> xr.DataArray:
    """Compute histogram of a single variable.

    Parameters
    ----------
    x
        The array to compute the histogram from.
    bins
        Bins specification that can be:

        * a Boost :external+boost-histogram:doc:`Axis<user-guide/axes>` object.
        * an :class:`int` for the number of bins in a
          :external+boost-histogram:ref:`/user-guide/axes.rst#regular-axis` where the
          minimum and maximum values are specified by *range* or computed from data.
    range
        The lower and upper range of the bins. If either is left to None, it will be
        computed with ``x.min()`` or ``x.max()``.
    weights
        Array of weights, broadcastable against the input *data*. Each value in *data*
        only contributes its associated weight towards the bin count (instead of 1). If
        density is False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.
    density
        If False (default), returns the number of samples in each bin. If True, returns
        the probability density function at the bin,
        ``bin_count / sample_count / bin_area``.
    dims
        Dimensions to compute the histogram along to. If left to None the data is
        flattened along all axes.
    storage: ~boost_histogram.storage.Storage
        Storage object used by the histogram. If None, the default one is used
        (:external+boost-histogram:ref:`/user-guide/storage.rst#double`). Currently,
        accumulator storage (with more than one value stored) are not supported.

    Returns
    -------
    histogram
        DataArray named ``<x name>_histogram``. The bins coordinates is named
        ``<x name>_bins``.
    """
    return histogramdd(
        x,
        bins=bins,
        range=[range] if range is not None else None,
        weights=weights,
        density=density,
        dims=dims,
        storage=storage,
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
    storage: bh.storage.Storage | None = None,
) -> xr.DataArray:
    """Compute 2-dimensional histogram.

    Parameters
    ----------
    x, y
        The arrays to compute the histogram from. They must be broadcastable against
        each other.
    bins
        Bins specification that can be:

        * a Boost :external+boost-histogram:doc:`Axis<user-guide/axes>` object.
        * an :class:`int` for the number of bins in a
          :external+boost-histogram:ref:`/user-guide/axes.rst#regular-axis` where the
          minimum and maximum values are specified by *range* or computed from data.

        If a single specification is passed, it will be reused for all variables.
        Otherwise a sequence of specification must be passed in the same order as the
        *x* and *y*.
    range
        Sequence of lower and upper ranges of the bins for each variable. If either is
        left to None, it will be computed with ``x.min()`` or ``x.max()``.
    weights
        Array of weights, broadcastable against the input *data*. Each value in *data*
        only contributes its associated weight towards the bin count (instead of 1). If
        density is True weights are normalized to 1. If density is False, the values of
        the returned histogram are equal to the sum of the weights belonging to the
        samples falling into each bin.
    density
        If False (default), returns the number of samples in each bin. If True, returns
        the probability density function at the bin,
        ``bin_count / sample_count / bin_area``.
    dims
        Dimensions to compute the histogram along to. If left to None the data is
        flattened along all axes.
    storage: ~boost_histogram.storage.Storage
        Storage object used by the histogram. If None, the default one is used
        (:external+boost-histogram:ref:`/user-guide/storage.rst#double`). Currently,
        accumulator storage (with more than one value stored) are not supported.

    Returns
    -------
    histogram
        DataArray named ``<x name>_<y name>_histogram``. The bins coordinates are named
        ``<variable name>_bins``.
    """
    return histogramdd(
        x,
        y,
        bins=bins,
        range=range,
        weights=weights,
        density=density,
        dims=dims,
        storage=storage,
    )


def histogramdd(
    *data: xr.DataArray,
    bins: abc.Sequence[BinsType] | BinsType = 10,
    range: abc.Sequence[RangeType] | None = None,
    weights: xr.DataArray | None = None,
    density: bool = False,
    dims: abc.Collection[abc.Hashable] | None = None,
    storage: bh.storage.Storage | None = None,
) -> xr.DataArray:
    """Compute N-dimensional histogram.

    Parameters
    ----------
    data
        The arrays to compute the histogram from. To compute a multi-dimensional
        histogram supply a sequence of as many arrays as the histogram dimensionality.
        Arrays must be broadcastable against each other. If any underlying data is a
        dask array, other inputs will be transformed into a dask array of a single
        chunk.
    bins
        Bins specification that can be:

        * a Boost :external+boost-histogram:doc:`Axis<user-guide/axes>` object.
        * an :class:`int` for the number of bins in a
          :external+boost-histogram:ref:`/user-guide/axes.rst#regular-axis` where the
          minimum and maximum values are specified by *range* or computed from data.

        If a single specification is passed, it will be reused for all variables.
        Otherwise a sequence of specification must be passed in the same order as the
        input *data*.
    range
        Sequence of lower and upper ranges of the bins for each variable. If either is
        left to None, it will be computed with ``x.min()`` or ``x.max()``.
    weights
        Array of weights, broadcastable against the input *data*. Each value in *data*
        only contributes its associated weight towards the bin count (instead of 1). If
        density is True weights are normalized to 1. If density is False, the values of
        the returned histogram are equal to the sum of the weights belonging to the
        samples falling into each bin.
    density
        If False (default), returns the number of samples in each bin. If True, returns
        the probability density function at the bin,
        ``bin_count / sample_count / bin_area``.
    dims
        Dimensions to compute the histogram along to. If left to None the data is
        flattened along all axes.
    storage: ~boost_histogram.storage.Storage
        Storage object used by the histogram. If None, the default one is used
        (:external+boost-histogram:ref:`/user-guide/storage.rst#double`). Currently,
        accumulator storage (with more than one value stored) are not supported.


    Returns
    -------
    histogram
        DataArray named ``<variables names separated by an underscore>_histogram``. The
        bins coordinates are named ``<variable name>_bins``.
    """
    variables = [a.name for a in data]
    bins_names = [f"{v}_bins" for v in variables]
    axes = get_axes_from_specs(bins, range, data)

    if storage is None:
        storage = bh.storage.Double()
    if type(storage) not in SIMPLE_STORAGE:
        warnings.warn(
            f"Accumulator storages are not supported (received {type(storage)})",
            UserWarning,
            stacklevel=1,
        )
    histref = bh.Histogram(*axes, storage=storage)

    if weights is not None:
        weights = weights.rename(VAR_WEIGHTS)
        data = data + (weights,)

    data = xr.broadcast(*data)

    if is_dask := is_any_dask(data):
        data = tuple(a.chunk({}) for a in data)
        data = xr.unify_chunks(*data)  # type: ignore[assignment]

        for ax in axes:
            if ax.traits.growth:
                raise ValueError(f"Axes cannot grow when using Dask ({ax})")

    data_dims = data[0].dims
    if dims is None:
        dims = data_dims
    # from collection to list in data order
    dims = [d for d in data_dims if d in dims]
    dims_loop = [d for d in data_dims if d not in dims]

    if is_dask:
        axis_loop = [data_dims.index(d) for d in dims_loop]
        axis_agg = [data_dims.index(d) for d in dims]
        counts = _histogram_dask(
            *[a.data for a in data],
            axis_loop=axis_loop,
            axis_agg=axis_agg,
            weight=weights is not None,
            histref=histref,
        )

        hist = xr.DataArray(counts, dims=dims_loop + bins_names, name=VAR_HIST)

    else:
        hist = xr.apply_ufunc(
            _blocked_dd,
            *data,
            input_core_dims=[list(dims) for _ in data],
            output_core_dims=[bins_names],
            vectorize=True,
            kwargs=dict(weight=weights is not None, histref=histref),
        ).rename(VAR_HIST)

    for name, b in zip(bins_names, axes, strict=True):
        hist = hist.assign_coords({name: b.edges[:-1]})
        hist[name].attrs["right_edge"] = b.edges[-1]

    if density:
        widths = [
            xr.DataArray(np.diff(b.edges), dims=[name])
            for b, name in zip(axes, bins_names, strict=True)
        ]
        areas = xr.DataArray(reduce(operator.mul, widths), dims=bins_names)
        hist = hist / areas / hist.sum(bins_names)

    hist_name = "pdf" if density else "histogram"
    hist = hist.rename("_".join(map(str, variables + [hist_name])))
    return hist


def _clone(histref: bh.Histogram) -> bh.Histogram:
    return bh.Histogram(*histref.axes, storage=histref.storage_type())


def _blocked_dd(
    *data: NDArray, weight: bool, histref: bh.Histogram, keepdims: bool = False
) -> NDArray:
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

    counts = thehist.values()

    if keepdims:
        counts = np.expand_dims(counts, tuple(_range(data[0].ndim)))

    return counts


def _blocked_dd_loop(
    *data: NDArray,
    axis_agg: abc.Sequence[int],
    axis_loop: abc.Sequence[int],
    weight: bool,
    histref: bh.Histogram,
    keepdims: bool = False,
) -> NDArray:
    """Multiple variables, ND arrays.

    Arrays are already broadcasted.
    """
    # move aggregated axes at the end
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

    counts = np.zeros((n_loop, *histref.shape))
    for i in range(n_loop):
        thehist = _clone(histref)
        if weight:
            *args, weights = flattened
            thehist.fill(*[a[i] for a in args], weight=weights[i])
        else:
            thehist.fill(*[a[i] for a in flattened])

        # TODO: take care of flow
        counts[i] = thehist.values()

    counts = np.reshape(counts, (*shape_loop, *histref.shape))
    if keepdims:
        counts = np.expand_dims(
            counts, tuple(_range(len(axis_loop), len(axis_loop) + len(axis_agg)))
        )

    return counts


def _histogram_dask(
    *data: da.Array,
    axis_loop: abc.Sequence[int],
    axis_agg: abc.Sequence[int],
    weight: bool,
    histref: bh.Histogram,
) -> da.Array:
    """Compute histogram for dask data."""
    if axis_loop:
        func = partial(
            _blocked_dd_loop,
            axis_loop=axis_loop,
            axis_agg=axis_agg,
            weight=weight,
            histref=histref,
        )
    else:
        func = partial(_blocked_dd, weight=weight, histref=histref)

    dtype = (
        int
        if histref.storage_type in (bh.storage.Int64, bh.storage.AtomicInt64)
        else float
    )
    # we don't use da.reduction. map_blocks allows for chunk changing shape,
    # and we use _tree_reduce to aggregate over axis_agg, this avoids the 'chunk' step
    # of da.reduction
    blocked = da.map_blocks(
        func,
        *data,
        keepdims=True,
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
    reduc = da.reductions._tree_reduce(
        blocked,
        da.sum,
        axis=axis_agg,
        keepdims=False,
        dtype=dtype,
        name="sum-hist",
        concatenate=True,
    )
    return reduc


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
                "Bins must be specified as boost Axis or with an integer "
                f"(received {type(spec)})."
            )

    return tuple(axes)


def is_any_dask(data: abc.Sequence[xr.DataArray]) -> bool:
    """Check if any the variables are in dask format.

    Only return true if Dask is imported.
    """
    return HAS_DASK and any(is_dask_collection(a.data) for a in data)
