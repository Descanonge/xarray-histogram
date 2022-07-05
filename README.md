
# XArray-Histogram

> Compute histograms from XArray data using BoostHistogram

This package allow to compute histograms from XArray data, taking advantage of
its label and dimensions management.
It relies on the [Boost Histogram](https://boost-histogram.readthedocs.io) library for the computations.

It is essentially a thin wrapper using directly [Boost Histogram](https://boost-histogram.readthedocs.io) on loaded data, or [Dask-histogram](https://dask-histogram.readthedocs.io) on data contained in dask arrays. It thus features optimised performance, as well as lazy computation and easy upscaling thanks to dask.

## Quick examples

Bins can be specified in a similarly to numpy functions:
``` python
import xarray_histogram as xh
hist = xh.histogram(data, bins=(100, 0., 10.))
```
but also can using boost [axis](https://boost-histogram.readthedocs.io/en/latest/user-guide/axes.html), benifiting from their features:
``` python
import boost_histogram as bh
hist = xh.histogram(data, bins=bh.axis.Regular(100, 0., 10.))
```

Multi-dimensional histogram can be computed, here in 2D for instance:
``` python
hist = xh.histogram(
    temp, sal,
    bins=[bh.axis.Regular(100, -5., 40.), bh.axis.Regular(100, 31, 38))
)
```

Finally, so far we have computed histograms on the whole flattened arrays, but we can compute only along some dimensions. For instance we can retrieve the time evolution of an histogram:
``` python
hist = xh.histogram(temp, bins=bh.axis.Regular(100, 0., 10.), dims=['lat', 'lon'])
```

Histograms can be normalised, and weights can be applied.
All of this works seamlessly with data stored in numpy or dask arrays.

## Requirements

- Python >= 3.7
- numpy
- xarray
- [boost-histogram](https://github.com/scikit-hep/boost-histogram)
- [dask](https://www.dask.org/) and [dask-histogram](https://github.com/dask-contrib/dask-histogram): Optional, if not available all data will be eagerly loaded.

## Documentation

Documentation and installation steps will be available on readthedocs.

For now install from source using `pip install -e .`.

## Tests and performance

Tests are on the way.
To compare performances check these notebooks for [numpy](./docs/source/perf_numpy.ipynb) and [dask](./docs/source/perf_dask.ipynb) arrays.

## Other packages

[xhistogram](https://xhistogram.readthedocs.io/en/latest/) already exists. It relies on numpy functions and thus does not benefit of some performance upgrades brought by Boost (see performance comparisons). I also hoped to bring similar features with simpler code, relying on dependencies. Some additional features of boost (overflow bins, rebinning) can easily be added (this is in the works).
