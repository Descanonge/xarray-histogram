
# XArray-Histogram

> Compute histograms from XArray data using BoostHistogram

This package allow to compute histograms from XArray data, taking advantage of
its label and dimensions management.
It relies on the [Boost Histogram](https://boost-histogram.readthedocs.io) library for the computations.

It is essentially a thin wrapper using directly [Boost Histogram](https://boost-histogram.readthedocs.io) on loaded data, or [Dask-histogram](https://dask-histogram.readthedocs.io) on data contained in dask arrays. It thus features optimised performance, as well as lazy computation and easy upscaling thanks to dask.

## Quick examples

Bins can be specified in a similar way to the numpy functions:
``` python
import xarray_histogram as xh
hist = xh.histogram(data, bins=(100, 0., 10.))
```
but can take advantage of boost [axis](https://boost-histogram.readthedocs.io/en/latest/user-guide/axes.html):
``` python
import boost_histogram as bh
hist = xh.histogram(data, bins=bh.axis.Regular(100, 0., 10.))
```

This works seamlessly on loaded data (stored in numpy arrays) or lazily loaded data (stored in dask arrays).

Multi-dimensional histogram can be computed, here 2D:
``` python
hist = xh.histogram(
    temp, sal,
    bins=[bh.axis.Regular(100, 0., 10.), bh.axis.Regular(100, 31, 38))
)
```

Finally, so far we have computed histograms on the whole flattened arrays, but we can compute only along some dimensions. For instance we can retrieve the time evolution of an histogram:
``` python
hist = xh.histogram(temp, bins=bh.axis.Regular(100, 0., 10.), dims=['lat', 'lon'])
```

Histograms can be normalised, and weights can be applied.

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

On its way as well.

## Other packages

[xhistogram](https://xhistogram.readthedocs.io/en/latest/) already exists and might suit you. It relies on numpy functions and thus does not benefit of some performance upgrades brought by Boost (see performance comparisons). I also hoped to bring similar features with a much simpler code. Some additional features of boost (overflow bins, rebinning) can easily be added (this is in the works).

It is also quite straightforward to compute an histogram from a flattened array with just boost: [Boost: Xarray example](https://boost-histogram.readthedocs.io/en/latest/notebooks/xarray.html), or just dask-histogram by using `dataarray.data.ravel()`.
