
# XArray-Histogram

> Compute histograms from XArray data using BoostHistogram

This package allow to compute histograms from XArray data, taking advantage of
its label and dimensions management.
It relies on the [Boost Histogram](https://boost-histogram.readthedocs.io) library for the computations.

It is essentially a thin wrapper using directly [Boost Histogram](https://boost-histogram.readthedocs.io) on loaded data, or [Dask-histogram](https://dask-histogram.readthedocs.io) on data contained in Dask arrays. It thus features optimized performance, as well as lazy computation and easy up-scaling thanks to Dask.

## Quick examples

Bins can be specified similarly to Numpy functions:
``` python
import xarray_histogram as xh
hist = xh.histogram(data, bins=[(100, 0., 10.)])
```
but also using boost [axes](https://boost-histogram.readthedocs.io/en/latest/user-guide/axes.html), benefiting from their features:
``` python
import boost_histogram.axis as bha
hist = xh.histogram(data, bins=[bha.Regular(100, 0., 10.)])
```

Multi-dimensional histogram can be computed, here in 2D for instance:
``` python
hist = xh.histogram(
    temp, chlorophyll,
    bins=[bha.Regular(100, -5., 40.), bha.Regular(100, 1e-3, 10, transform=bha.transform.log))
)
```

Finally, so far we have computed histograms on the whole flattened arrays, but we can compute only along some dimensions. For instance we can retrieve the time evolution of an histogram:
``` python
hist = xh.histogram(temp, bins=[bha.Regular(100, 0., 10.)], dims=['lat', 'lon'])
```

Histograms can be normalized, and weights can be applied.
All of this works seamlessly with data stored in numpy or dask arrays.

## Requirements

- Python >= 3.11
- numpy
- xarray
- [boost-histogram](https://github.com/scikit-hep/boost-histogram)
- [dask](https://www.dask.org/) and [dask-histogram](https://github.com/dask-contrib/dask-histogram) (optional)

## Installation

From source with:
``` shell
git clone https://github.com/Descanonge/xarray-histogram
cd xarray-histogram
pypi install -e .
```

Soon on Pypi.

## Documentation

Documentation available at https://xarray-histogram.readthedocs.io

## Installation

Soon from PyPI ... 🚧

From source:
``` shell
git clone https://github.com/Descanonge/xarray-histogram
cd xarray-histogram
pypi install -e .
```

## Tests and performance

To compare performances check these notebooks for [numpy](./docs/source/perf_numpy.ipynb) and [dask](./docs/source/perf_dask.ipynb) arrays.

## Other packages

[xhistogram](https://xhistogram.readthedocs.io/en/latest/) already exists. It relies on Numpy functions and thus does not benefit of some performance upgrades brought by Boost (see performance comparisons).
I also hoped to bring similar features with simpler code, relying on dependencies. Some additional features of boost (overflow bins, rebinning, extracting various statistics from the DataArray histogram) could be added (this is in the works).
