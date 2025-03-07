#!/usr/bin/env python3

from collections.abc import Callable, Sequence
from typing import Any

import boost_histogram as bh
import dask.array as da
import numpy as np
import pytest
import xarray as xr


def get_array(
    shape: Sequence[int], chunks: Sequence[int] | None = None, name: str = "var1"
) -> xr.DataArray:
    if chunks is None:
        data = np.random.rand(*shape)
    else:
        data = da.random.random(shape, chunks=chunks)

    return xr.DataArray(data, dims=list("xywzuv"[: len(shape)]), name=name)


def get_weights(x: xr.DataArray, weight: bool) -> xr.DataArray | None:
    if not weight:
        return None
    chunks = [c[0] for c in x.chunks] if x.chunks else None
    return get_array(x.shape, chunks=chunks, name="weights")


def get_ref_hist(
    *data: xr.DataArray,
    axes: Sequence[bh.axis.Axis],
    weights: xr.DataArray | None = None,
    **kwargs,
) -> np.ndarray:
    # All inputs must have the same shape!!
    args: Any = [x.compute().data.reshape(-1) for x in data]
    kwargs["bins"] = [ax.edges for ax in axes]
    kwargs["weights"] = (
        weights.compute().data.reshape(-1) if weights is not None else None
    )

    if len(args) == 1:
        func: Callable = np.histogram
        args = args[0]
        kwargs["bins"] = kwargs["bins"][0]
    else:
        func = np.histogramdd

    ref, _ = func(args, **kwargs)
    return ref


def bool_param(name: str):
    values = [True, False]
    ids = [f"{name}:{v}" for v in values]
    return pytest.mark.parametrize(name, values, ids=ids)


def id_x(x: xr.DataArray):
    s = f"x{x.shape}"
    if x.chunks is not None:
        s += str([d[0] for d in x.chunks])
    return s


def id_data(data: Sequence[xr.DataArray]) -> str:
    return f"{{{':'.join(id_x(x) for x in data)}}}"
