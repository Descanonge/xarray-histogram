
# XArray-Histogram

> Compute and manipulate histograms from XArray data using BoostHistogram

<div align="left">

[![PyPI](https://img.shields.io/pypi/v/xarray-histogram)](https://pypi.org/project/xarray-histogram)
[![GitHub release](https://img.shields.io/github/v/release/Descanonge/xarray-histogram)](https://github.com/Descanonge/xarray-histogram/releases)
![test status](https://github.com/Descanonge/xarray-histogram/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/xarray-histogram/badge/?version=latest)](https://xarray-histogram.readthedocs.io/en/latest/?badge=latest)

</div>

This package allows to compute histograms from and to XArray data.
It relies on the [Boost Histogram](https://boost-histogram.readthedocs.io) library giving better performances compared to `numpy.histogram` and the existing [xhistogram](https://xhistogram.readthedocs.io/en/latest/).
It also brings features such as integer/discrete bins or periodic bins.

Dask arrays are supported.

Vectorized manipulation and analysis of the resulting histogram(s) is provided via an XArray accessor.

## Quick examples

Three functions are provided (histogram, histogram2d, and historamdd), similar to those from Numpy:
``` python
import xarray_histogram as xh
hist = xh.histogram(data, bins=100, range=(0, 10))
```

Bins can be specified directly via Boost [axes](https://boost-histogram.readthedocs.io/en/latest/user-guide/axes.html) for a finer control. The equivalent of the example above would be:
``` python
import boost_histogram.axis as bha
hist = xh.histogram(data, bins=[bha.Regular(100, 0., 10.)])
```

Multi-dimensional histogram can be computed, here in 2D for instance:
``` python
hist = xh.histogramdd(
    temp, chlorophyll,
    bins=[bha.Regular(100, -5., 40.), bha.Regular(100, 1e-3, 10, transform=bha.transform.log))
)
```

The histograms can be computed on the whole flattened arrays, but we can apply it to only some dimensions. For instance if we have an array of dimensions `(time, lat, lon)` we can retrieve the time evolution of its histogram:
``` python
hist = xh.histogram(temp, bins=[bha.Regular(100, 0., 10.)], dims=['lat', 'lon'])
```

Weights can be applied. Output histogram can be normalized

## Accessor

An Xarray [accessor](https://docs.xarray.dev/en/latest/internals/extending-xarray.html) is provided to do some vectorized manipulations on histogram data. Simply import `xarray_histogram.accessor`, and all arrays can then access methods through the `hist` property::

``` python
import xarray_histogram.accessor

hist = xh.histogram(temp, ...)

hist.hist.edges()
hist.hist.median()
hist.hist.ppf(q=0.75)
```

See the [documentation](https://xarray-histogram.readthedocs.io/en/latest/accessor.html) for more details.

## Documentation

Documentation available at https://xarray-histogram.readthedocs.io

## Installation

From PyPI:

``` shell
pip install xarray-histogram
```

From source:
``` shell
git clone https://github.com/Descanonge/xarray-histogram
cd xarray-histogram
pip install -e .
```

## TODO

Some features of Boost are not yet available:
- Growing axes: Dask requires to know in advance the size of output chunks. This could reasonably be supported, at least when applying over the whole array (no looping dimensions).
- Advanced storage/accumulators: they provide additional values on top of the count of samples falling into a bin. They require more than one number per bin, and a more complex sum of two histograms (possibly making histogram along chunked dimensions impossible). 
- The [Unified Histogram Indexing](https://uhi.readthedocs.io/en/latest/indexing.html) could be implemented in the accessor to facilitate manipulation of histogram arrays.

## Requirements

- Python >= 3.11
- numpy
- xarray
- [boost-histogram](https://github.com/scikit-hep/boost-histogram)
- [dask](https://www.dask.org/) (optional)
- scipy (optional, for accessor)

## Tests and performance

To compare performances check this [notebook](./docs/source/performances.ipynb).

## Other packages

[xhistogram](https://xhistogram.readthedocs.io/en/latest/) already exists. It relies on Numpy functions ([searchsorted](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html)) and thus does not benefit of some performance upgrades brought by Boost (see performance comparisons).

[dask-histogram](https://github.com/dask-contrib/dask-histogram) ports Boost-histogram for Dask. It does not support multi-dimensional arrays: one can still reshape the input array but this can incur [performance penalties](https://docs.dask.org/en/stable/array-chunks.html#reshaping). Still, as it works directly with boost objects rather than Dask arrays all features of Boost should be available.
