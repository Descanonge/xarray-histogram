
.. currentmodule:: xarray_histogram

Usage
=====

This package supplies a single function to compute histograms, whatever their
dimensionality: :func:`~core.histogram`.

Input data
----------

Its first parameter is the :class:`DataArrays<xarray.DataArray>` on which to
compute the histogram. The function also accept an optional argument giving the
weights to apply when computing the histogram. A single array will give a 1-d
histogram, two arrays a 2-d histogram, etc.

The underlying data can be Numpy arrays, in which case we will use
:class:`boost_histogram.Histogram`; or Dask arrays for which we will use
:func:`dask_histogram.partitioned_factory`. The latter does not accepts Numpy
arrays, so in case of mixed types input (this include the weights) each non-Dask
array will be transformed in a Dask array of a single chunk.

The different arrays must be broadcastable against each other. For Dask arrays,
chunks will be adapted using :func:`xarray.unify_chunks`.

.. note::

   You may have noticed we do not use :func:`dask_histogram.factory`. This is
   because (perhaps surprisingly) it merges all chunks along histogram
   dimensions before computing. This can result in loading in memory larger
   amount of data than one might have hoped. Instead, we create a
   :class:`~dask_histogram.PartitionedHistogram` (one histogram per chunk), and
   then aggregate the results into one histogram using
   :meth:`dask_histogram.PartitionedHistogram.collapse`.


Bins / Axes
-----------

To specify the bins for the different variables, a sequence of
:data:`specifications<core.AxisSpec>` *of the same length as the histogram
dimensionality* must be given to the `bins` argument.

A specification can be an :class:`boost_histogram.axis.Axis` instance. This is
used by the boost library, it allows for optimization and features like under
and overflow bins and more. Some basic examples of axis include::

   import boost_histogram.axis as bha

   # regular width bins
   bha.Regular(200, 0., 10.)
   # logarithmically spaced bins (without performance loss)
   bha.Regular(200, 1e-3, 10., transform=bha.transform.log)
   # integer bins
   bha.Integer(0, 20)

For more details on creating axes, see the user guide on
:external+boost-histogram:doc:`user-guide/axes`.

A specification can be a sequence of three numbers: the number of bins, the
minimum value, and maximum value. It will result in a regular axis::

  spec = (nbins, vmin, vmax)
  axis = bha.Regular(nbins, vmin, vmax)


Lastly, a specification can also be a single integer denoting the number of
bins, the minimum and maximum values will be computed from the data. This is
discouraged since the computation will be triggered on the spot. A warning will
be emitted, to silence it execute :func:`~core.silent_minmax_warning`
beforehand.::

  spec = nbins
  axis = bha.Regular(nbins, float(variable.vmin()), float(variable.vmax()))


Output
------

For now, :func:`~core.histogram` returns a simple :class:`xarray.DataArray`.


Examples
========

Simple histogram::

  from xarray_histogram import histogram
  import boost_histogram.axis as bha

  hist = histogram(temp, bins=[bha.Regular(100, -5., 40.)])
  hist.plot.line()

Multi-dimensional histogram, here in 2D for instance::

   hist = histogram(
      temp, chlorophyll,
      bins=[
         bh.Regular(100, -5., 40.),
         bh.Regular(100, 1e-3, 20, transform=bha.transform.log)
      ]
   )
   hist.plot.pcolormesh()

Finally, so far we have computed histograms on the whole flattened arrays, but
we can compute only along some dimensions. For instance we can retrieve the time
evolution of an histogram::

   hist = histogram(
      temp,
      bins=[bha.Regular(100, 0., 10.)],
      dims=['lat', 'lon']
   )
   hist.plot.line(x="temp_bins")
