
from typing import (
    Any,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Union
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


class BinsMinMaxWarning(UserWarning):
    pass


def histogram(
        data: Union[xr.DataArray, Sequence[xr.DataArray]],
        bins: Union[bh.axis.Axis, Dict[Hashable, bh.axis.Axis]],
        dims: Optional[Sequence[Hashable]] = None,
        weight: Union[None, xr.DataArray] = None,
        density: bool = False
):
    """Compute histogram of a DataArray.

    Parameters
    ----------
    data:
    bins: Mappable of dimension name to a boost-histogram axis.
    dims:
    weight:
    density:
    """
    data = to_list(data)
    data_sanity_check(data)
    variables = [a.name for a in data]

    if not isinstance(bins, dict):
        bins = {variables[0]: bins}
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

    # Merge everything together so it can be send through a single
    # groupby call.
    ds = xr.merge(data, join='exact')

    do_flat_array = (dims is None or set(dims) == set(data[0].dims))

    if all_dask:
        comp_hist_func = comp_hist_dask
    else:
        comp_hist_func = comp_hist_numpy


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
        # from xhistogram
        if len(widths) == 1:
            area = widths[0]
        elif len(widths) == 2:
            area = np.outer(*widths)
        else:
            area = np.prod(np.ix_(*widths))

        area = xr.DataArray(area, dims=bins_names)
        hist = hist / area / hist.sum(bins_names)

    return hist


def separate_ravel(ds, variables):
    data = [ds[v].data.ravel() for v in variables]
    if VAR_WEIGHT in ds:
        weight = ds[VAR_WEIGHT].data.ravel()
    else:
        weight = None
    return data, weight


def post_comp(h_values, bins_names):
    return xr.DataArray(h_values, dims=bins_names)


def comp_hist_dask(ds, variables, bins, bins_names):
    data, weight = separate_ravel(ds, variables)
    hist = dh.factory(*data, axes=bins, weights=weight)
    h_values, _ = hist.to_dask_array()
    return post_comp(h_values, bins_names)


def comp_hist_numpy(ds, variables, bins, bins_names):
    hist = bh.Histogram(*bins)
    data, weight = separate_ravel(ds, variables)
    hist.fill(*data, weight=weight)
    return post_comp(hist.values(), bins_names)


def to_list(a: Union[Any, Sequence]) -> Sequence:
    """Put argument as list of not already a Sequence."""
    if not isinstance(a, Sequence):
        return [a]
    return a


def data_sanity_check(data: Sequence[xr.DataArray]):
    if len(data) == 0:
        raise TypeError("Data sequence of length 0.")
    dims0 = set(data[0].dims)
    for a in data:
        if not isinstance(a, xr.DataArray):
            raise TypeError("Data must be a xr.DataArray, "
                            f"a type {type(a).__name__} was supplied.")
        if set(a.dims) != dims0:
            raise ValueError("Dimensions are different in supplied arrays.")


def weight_sanity_check(weight, data):
    dims0 = set(data[0].dims)
    if not isinstance(weight, xr.DataArray):
        raise TypeError("Weights must be a xr.DataArray, "
                        f"a type {type(weight).__name__} was supplied.")
    if set(weight.dims) != dims0:
        raise ValueError("Dimensions are different in supplied weights.")


def silent_minmax_warning():
    warnings.filterwarnings("ignore", category=BinsMinMaxWarning)


def manage_bins_input(bins, data):
    if len(bins) != len(data):
        raise ValueError(f"Not as much bins specifications ({len(bins)}) "
                         f"as data arrays ({len(data)}) were supplied")
    data_dict = {a.name: a for a in data}
    bins_dict = {}
    for name, spec in bins.items():
        if name not in data_dict:
            raise KeyError(f"{name} bins name does not correspond to any"
                           "given DataArray.")
        a = data_dict[name]
        if isinstance(spec, bh.axis.Axis):
            bins_dict[name] = spec
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
        bins_dict[name] = bh.axis.Regular(*spec)

    bins_out = [bins_dict[a.name] for a in data]
    return bins_out


def has_all_dask(data) -> bool:
    all_dask = HAS_DASK and all(isinstance(a.data, dsa.core.Array)
                                for a in data)
    return all_dask
