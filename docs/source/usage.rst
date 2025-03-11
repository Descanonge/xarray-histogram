
.. currentmodule:: xarray_histogram

*****
Usage
*****

This package supplies functions similar to those of numpy:
:func:`~core.histogram`, :func:`~core.histogram2d` and
:func:`~core.histogramdd`.

Input data
==========

The first parameters are the :class:`DataArray(s)<xarray.DataArray>` on which to
compute the histogram. The function also accept an optional argument giving the
weights to apply when computing the histogram. All arguments must be
broadcastable against each other. For Dask arrays, chunks will be adapted using
:func:`xarray.unify_chunks`.

Bins / Axes
===========

The bins can be specified by either:

* a :external+boost-histogram:doc:`Boost Axis<user-guide/axes>` object for finer
  control
* an :class:`int` giving the number of bins. The minimum and maximum values are
  specified with the ``range`` argument or computed from data. The axis will be
  :external+boost-histogram:ref:`Regular </user-guide/axes.rst#regular-axis>`.

The ``range`` argument can supply the minimum and maximum values. Either or both
can be set to None in which case it will be computed with ``x.min()`` or
``x.max()``.
So for instance::

    xh.histogram(x, bins=10, range=(0., None))

will result in a regular axis (equal width bins) ranging from 0 to the maximum
value in *x*.

Directly passing an array of edges as in :func:`numpy.histogram` is not
supported. Instead, use a
:external+boost-histogram:ref:`/user-guide/axes.rst#variable-axis`.

.. tip::

   Using regularly spaced bins (even with a
   :external+boost-histogram:ref:`transform
   </user-guide/axes.rst#regular-axis-transforms>` applied) is more efficient:
   it avoids having to use binary search to find in which bin a value falls.

Some basic examples of axis include::

   import boost_histogram.axis as bha

   # regular width bins
   bha.Regular(200, 0., 10.)
   # logarithmically spaced bins (without performance loss)
   bha.Regular(200, 1e-3, 10., transform=bha.transform.log)
   # integer bins
   bha.Integer(0, 20)
   # boolean
   bha.Integer(0, 2, underflow=False, overflow=True)

Over/underflow
==============

By default, Boost axes are configured to keep count of the data points that
fall outside their range. Pass ``underflow=False`` and/or ``overflow=False``
when creating an axis to disable this.
Still by default, the flow bins values are not kept in the output array.

To keep the flow bins, pass ``flow=True`` to the histogram functions. The
coordinates values for the underflow and overflow bins will be set to

- for a float variable: :data:`-np.inf<numpy.inf>` and :data:`np.inf<numpy.inf>`
- for an integer variable: the minimum and maximum values of the dtype


Output
======

All three functions return a simple :class:`xarray.DataArray`. Its name is
``<variable names separated by underscores>_histogram`` (so for instance
``x_y_histogram``). The bins edges are contained in coordinates named
``<variable>_bins``. The right edge of the last bin is stored in a coordinate
attribute.

The nomenclature is the same as :external+xhistogram:doc:`xhistogram <index>` to
ensure easy transition between the two packages. It also enables the use of an
:doc:`accessor <accessor>` for extra features.

The dtype of output DataArray is ``int`` if the
:external+boost-histogram:doc:`storage <user-guide/storage>` is one of
:class:`~boost_histogram.storage.Int64` or
:class:`~boost_histogram.storage.AtomicInt64`, or ``float`` otherwise.

.. admonition:: 🚧

    It could be possible to enforce a given dtype. TODO...

Examples
========

Simple histogram::

  import xarray_histogram as xh
  import boost_histogram.axis as bha

  hist = xh.histogram(temp, bins=bha.Regular(100, -5., 40.))
  hist.plot.line()

Multi-dimensional histogram, here in 2D for instance::

   hist = xh.histogram2d(
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

   hist = xh.histogram(
      temp,
      bins=bha.Regular(100, 0., 10.),
      dims=['lat', 'lon']
   )
   hist.plot.line(x="temp_bins")
