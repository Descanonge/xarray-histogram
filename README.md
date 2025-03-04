
# XArray-Histogram

> Compute and manipulate histograms from XArray data using BoostHistogram

This package allow to compute histograms from and to XArray data.
It relies on the [Boost Histogram](https://boost-histogram.readthedocs.io) library giving better performances compared to Numpy.histogram and the existing [xhistogram](https://xhistogram.readthedocs.io/en/latest/).
It also brings features such as integer/discrete bins or periodic bins.

Dask arrays are supported.

Vectorized manipulation and analysis of the resulting histogram(s) is provided via an XArray accessor.

## Quick examples

Three functions similar as Numpy are provided (histogram, histogram2d, and historamdd).
``` python
import xarray_histogram as xh
hist = xh.histogram(data, bins=100, range=(0, 10))
```

Bins can be specified directly via Boost [axes](https://boost-histogram.readthedocs.io/en/latest/user-guide/axes.html) for a finer control. The equivalent would be:
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

The histograms can be computed on the whole flattened arrays, but we can apply it to only some dimensions. For instance we can retrieve the time evolution of an histogram:
``` python
hist = xh.histogram(temp, bins=[bha.Regular(100, 0., 10.)], dims=['lat', 'lon'])
```

Weights can be applied. Output histogram can be normalized

## Accessor

An Xarray [accessor](https://docs.xarray.dev/en/latest/internals/extending-xarray.html) is provided to do some vectorized manipulations on histogram data. Simply import `xarray_histogram.accessor`, and all arrays can then access methods through the `hist` property::

``` python
hist.hist.edges()
hist.hist.median()
hist.hist.ppf(q=0.75)
```

See the [accessor API](https://xarray-histogram.readthedocs.io/en/latest/_api/xarray_histogram.accessor.html) for more details.

The [Unified Histogram Indexing](https://uhi.readthedocs.io/en/latest/indexing.html) could be implemented in the accessor, especially for the rebinning operations.

## Dask support

Dask arrays are supported for most use cases, however some features of Boost are not yet available:
- Growing axes: Dask requires to know in advance the size of output chunks. This could reasonably be supported, at least when applying over the whole array (no looping dimensions).
- Advanced storage: they make available various variables such as the variance. They require more than one number per bin, and a more complex sum of two histograms (possibly making histogram along chunked dimensions impossible). 

## Requirements

- Python >= 3.11
- numpy
- xarray
- [boost-histogram](https://github.com/scikit-hep/boost-histogram)
- [dask](https://www.dask.org/) (optional)
- scipy (optional, for accessor)

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

[dask-histogram](https://github.com/dask-contrib/dask-histogram) ports Boost-histogram for Dask. It does not support multi-dimensional arrays, but outputs boost objects directly rather than Dask arrays.
