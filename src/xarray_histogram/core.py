"Main functions."

# This file is part of the 'xarray-histogram' project
# (http://github.com/Descanonge/xarray-histogram) and subject
# to the MIT License as defined in the file 'LICENSE',
# at the root of this project. © 2022 Clément Haëck

from __future__ import annotations
from typing import (
    Any,
    Hashable,
    List,
    Sequence,
    Union,
    Tuple
)
import warnings

import boost_histogram as bh
import numpy as np
import xarray as xr

try:
    import dask.array as dsa
    import dask_histogram as dh
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


VAR_WEIGHT = '_weight'

AxisSpec = bh.axis.Axis | int | Sequence[int | float]


class BinsMinMaxWarning(UserWarning):
    pass


def histogram_varbins(data, bins, dims=None, weight=None, density=None):
    data = to_list(data)
    data_sanity_check(data)
    variables = [a.name for a in data]

    bins = to_list(bins)
    for i, (b, v) in enumerate(zip(bins, variables)):
        bin_dim = set(b.dims) - set(data[0].dims)
        if len(bin_dim) == 0:
            raise KeyError(f"No bins dimension found for {v} bins.")
        if len(bin_dim) > 1:
            raise KeyError(f"Two bins dimensions found for {v} bins.")
        bins[i] = b.rename({list(bin_dim)[0]: '_bins'}).rename(f'bins_{v}')

    bins_names = [b.name for b in bins]

    bins_dims = set(bins[0].dims) & set(data[0].dims)
    data_dims = set(data[0].dims)

    if dims is None:
        dims = data_dims

    for b in bins:
        if bins_dims & set(dims):
            raise KeyError('bins dims cannot be flattened')

    all_dask = has_all_dask(data)
    if not all_dask:
        for i, a in enumerate(data):
            if not a._in_memory:
                data[i] = a.load()

    ds = xr.merge(data + bins, join='exact')

    do_flat_array = (set(dims) == data_dims)
    if not do_flat_array:
        stacked_dim1 = list(bins_dims)
        stacked_dim2 = data_dims - set(dims) - bins_dims
        print(stacked_dim2)
        ds_stack1 = ds.stack(stacked_dim1=stacked_dim1)

        ds_int = ds_stack1.groupby('stacked_dim1').map(
            comp_, shortcut=True,
            args=[variables, bins_names, stacked_dim2]
        )

        out = ds_int.unstack()
        return out

    return ds


def comp_(ds, variables, bins_names, stacked_dim):
    bins = [bh.axis.Variable(ds[b]) for b in bins_names]
    out = ds.stack(stacked_dim2=stacked_dim).groupby('stacked_dim2').map(
        comp_hist_numpy, shortcut=True,
        args=[variables, bins, bins_names]
    )
    return out


def histogram(
        data: xr.DataArray | Sequence[xr.DataArray],
        bins: AxisSpec | Sequence[AxisSpec],
        dims: Sequence[Hashable] | None = None,
        weight: xr.DataArray | None = None,
        density: bool = False
):
    """Compute histogram.

    Parameters
    ----------
    data : DataArray or sequence of DataArray
        The `xarray.DataArray` to compute the histogram from. To compute a
        multi-dimensional histogram supply a sequence of as many arrays
        as the histogram dimensionality. All arrays must have the same
        dimensions.
    bins : Axis specification or sequence of axis specification, optional
        Specification of the histograms bins. Supply a sequence of
        specifications for a multi-dimensional array, in the same order as
        the `data` arrays.
        Specification can either be: a `boost-histogram.axis.Axis` or subtype.
        It can also be a tuple of (number of bins, minimum value,
        maximum value) in which case the bins will be linearly spaced.
        Only the number of bins can be specified, the min-max values are then
        computed on the spot.
    dims : sequence of str, optional
        Dimensions to compute the histogram along to. If left to None the
        data is flattened along all axis.
    weight : DataArray, optional
        `xarray.DataArray` of the weights to apply for each data-point.
        It must have the same dimensions as the data arrays.
    density : bool, optional
        If true normalize the histogram so that its integral is one.
        Does not take into account `weight`. Default is false.


    Returns
    -------
    histogram : DataArray
        DataArray named `hist_<variable name>` or just `hist` for
        multi-dimensional histograms. The bins coordinates are named
        `bins_<variable name>`.
    """
    data = to_list(data)
    data_sanity_check(data)
    variables = [a.name for a in data]

    bins = to_list(bins)
    bins = manage_bins_input(bins, data)
    bins_names = ['bins_' + v for v in variables]

    if weight is not None:
        weight_sanity_check(weight, data)
        weight = weight.rename(VAR_WEIGHT)
        data.append(weight)

    all_dask = has_all_dask(data)
    if not all_dask:
        for i, a in enumerate(data):
            if not a._in_memory:
                data[i] = a.load()

    # Merge everything together so it can be sent through a single
    # groupby call.
    ds = xr.merge(data, join='exact')

    if all_dask:
        comp_hist_func = comp_hist_dask
    else:
        comp_hist_func = comp_hist_numpy

    do_flat_array = (dims is None or set(dims) == set(data[0].dims))
    if do_flat_array:
        hist = comp_hist_func(ds, variables, bins, bins_names)
    else:
        stacked_dim = [d for d in data[0].dims if d not in dims]
        ds_stacked = ds.stack(stacked_dim=stacked_dim)

        hist = ds_stacked.groupby("stacked_dim").map(
            comp_hist_func, shortcut=True,
            args=[variables, bins, bins_names]
        )
        hist = hist.unstack()

    hist = hist.rename('hist')
    for name, b in zip(bins_names, bins):
        hist = hist.assign_coords({name: b.edges[:-1]})
        hist[name].attrs['right_edge'] = b.edges[-1]

    if density:
        widths = [np.diff(b.edges) for b in bins]
        if len(widths) == 1:
            area = widths[0]
        elif len(widths) == 2:
            area = np.outer(*widths)
        else:
            # https://stackoverflow.com/a/43149109
            # I added dtype=object to silence a warning for ragged arrays
            # not sure how safe this is.
            area = np.prod(np.array(np.ix_(*widths), dtype=object))

        area = xr.DataArray(area, dims=bins_names)
        hist = hist / area / hist.sum(bins_names)

    return hist


def separate_ravel(
        ds: xr.DataSet, variables: Sequence[str]
) -> Tuple[List[xr.DataArray], xr.DataArray]:
    """Separate data and weight arrays and flatten arrays."""
    data = [ds[v].data.ravel() for v in variables]
    if VAR_WEIGHT in ds:
        weight = ds[VAR_WEIGHT].data.ravel()
    else:
        weight = None
    return data, weight


def post_comp(h_values, bins_names):
    """Create an histogram DataArray."""
    return xr.DataArray(h_values, dims=bins_names)


def comp_hist_dask(
        ds: xr.Dataset,
        variables: Sequence[str],
        bins: Sequence[bh.axis.Axis],
        bins_names: Sequence[str]
) -> xr.DataArray:
    """Compute histogram for dask data."""
    data, weight = separate_ravel(ds, variables)
    hist = dh.factory(*data, axes=bins, weights=weight)
    res = hist.to_dask_array()
    return post_comp(res[0], bins_names)


def comp_hist_numpy(
        ds: xr.Dataset,
        variables: Sequence[str],
        bins: Sequence[bh.axis.Axis],
        bins_names: Sequence[str]
) -> xr.DataArray:
    """Compute histogram for numpy data."""
    hist = bh.Histogram(*bins)
    data, weight = separate_ravel(ds, variables)
    hist.fill(*data, weight=weight)
    return post_comp(hist.values(), bins_names)


def to_list(a: Union[Any, Sequence]) -> Sequence:
    """Put argument as list of not already a Sequence."""
    if a is None:
        return []
    if not isinstance(a, Sequence):
        return [a]
    return a


def data_sanity_check(data: Sequence[xr.DataArray]):
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
            raise TypeError("Data must be a xr.DataArray, "
                            f"a type {type(a).__name__} was supplied.")
    dims0 = set(data[0].dims)
    for a in data:
        if set(a.dims) != dims0:
            raise ValueError("Dimensions are different in supplied arrays.")


def weight_sanity_check(weight, data):
    """Ensure weight is correctly formated.

    Raises
    ------
    TypeError: If weight is not a `xarray.DataArray`.
    ValueError: If the set of dimensions are not the same in weights as data.
    """
    dims0 = set(data[0].dims)
    if not isinstance(weight, xr.DataArray):
        raise TypeError("Weights must be a xr.DataArray, "
                        f"a type {type(weight).__name__} was supplied.")
    if set(weight.dims) != dims0:
        raise ValueError("Dimensions are different in supplied weights.")
    # We only check for correct set of dimensions. Weight will be aligned
    # later with data anyway


def silent_minmax_warning():
    """Filter out the warning for computing min-max values."""
    warnings.filterwarnings("ignore", category=BinsMinMaxWarning)


def manage_bins_input(
        bins: Sequence[AxisSpec],
        data: Sequence[xr.DataArray]
) -> Sequence[bh.axis.Axis]:
    """Check bins input and convert to boost objects.

    Raises
    ------
    ValueError: If there are not as much bins specifications as data arrays.
    """
    if len(bins) != len(data):
        raise ValueError(f"Not as much bins specifications ({len(bins)}) "
                         f"as data arrays ({len(data)}) were supplied")
    bins_out = []
    for spec, a in zip(bins, data):
        if isinstance(spec, bh.axis.Axis):
            bins_out.append(spec)
            continue
        if isinstance(spec, int):
            spec = [spec]
        if len(spec) < 3:
            warnings.warn(
                ("Range was not supplied, the minimum and maximum values "
                 "will have to be computed. use `silent_minmax_warning` to "
                 "ignore this warning."),
                BinsMinMaxWarning)
            spec = [spec[0], float(a.min().values),
                    float(a.max().values)]
        bins_out.append(bh.axis.Regular(*spec))

    return bins_out


def has_all_dask(data) -> bool:
    """Check if all data is in dask format.

    And if dask-histogram is available.
    """
    all_dask = HAS_DASK and all(isinstance(a.data, dsa.core.Array)
                                for a in data)
    return all_dask
