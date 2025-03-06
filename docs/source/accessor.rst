.. currentmodule:: xarray_histogram

********
Accessor
********

An :external+xarray:doc:`accessor <internals/extending-xarray>` is provided to
ease manipulation and analysis of the histogram outputs. Simply import
:mod:`xarray_histogram.accessor` to register it. It will then be available for
all DataArrays that meet some conditions (:ref:`see below
<accessor-conditions>`), under the ``hist`` attribute. It gives access to a
number of methods. ::

    import xarray_histogram as xh
    import xarray_histogram.accessor

    h = xh.histogram(data)

    h.hist.median()

Operations are vectorized, so that you can apply them to entire arrays of
histograms. For instance for data defined along time, latitude and longitude,
we can compute one histogram per time-step::

    >>> h = xh.histogram(data, dims=["lon", "lat"])
    >>> h.hist.median()
    will be of dimensions ("time",)

Computations
============

Bins
----

The accessor provides the bins edges as a DataArray of size N+1 (it includes the
last bins right edge) for a given variable:
:meth:`~.HistDataArrayAccessor.edges`. Similarly, it provides the bins
:meth:`~.HistDataArrayAccessor.centers`, :meth:`~.HistDataArrayAccessor.widths`,
and :meth:`~.HistDataArrayAccessor.areas`.

Normalization
-------------

.. important::

   The accessor considers the histogram normalized or not given the name of its
   DataArray: normalized if named ``<variables>_pdf`` and non-normalized
   if ``<variables>_histogram``. This is consistent with the output of
   :func:`~.core.histogram`.

The histogram can be normalized if not already, using
:meth:`~.HistDataArrayAccessor.normalize`. Note that for a N-dimensional
histogram, this function can normalize only some variables.

Bins transform
--------------

Arbitrary functions can be applied to bins with
:meth:`~.HistDataArrayAccessor.apply_func`. The result is equivalent to
computing an histogram of ``func(data["variable"])``. The function must
transform the N+1 edges given as a DataArray. There is no need to account for
the *right_edge* attribute.

For instance, :meth:`~.HistDataArrayAccessor.scale` scales bins by a given
factor. It essential does ``hist.apply_func(lambda edges: edges * factor)``

Statistics
----------

A number of statistics can be extracted from the histogram. The following
functions are wrappers around methods of :class:`scipy.stats.rv_histogram`.

.. note::

    The histogram cannot be chunked in any bins dimensions.

.. autosummary::

   ~accessor.HistDataArrayAccessor.cdf
   ~accessor.HistDataArrayAccessor.interval
   ~accessor.HistDataArrayAccessor.median
   ~accessor.HistDataArrayAccessor.mean
   ~accessor.HistDataArrayAccessor.moment
   ~accessor.HistDataArrayAccessor.ppf
   ~accessor.HistDataArrayAccessor.std
   ~accessor.HistDataArrayAccessor.var


.. _accessor-conditions:

Conditions of accessibility
===========================

Once registered, an accessor is a cached property that can be accessed on any
DataArray. They are some conditions for the *hist* accessor to be created
successfully:

* The coordinates of the bins must be named ``<variable>_bins``.
* Each bins coordinates must contain an attribute named ``right_edge``,
  corresponding to the right edge of the last bin.
* The array must be named as ``<variable(s)_name>_<histogram or pdf>``.
  *histogram* if it is not normalized, and *pdf* if it is normalized as a
  probability density function. If the histogram is multi-dimensional, the
  variables names must be separated by underscores. For instance:
  ``Temp_Sal_histogram``.

Those conventions are coherent with the output of
``xarray_histogram.histogram*``, so if you use this packages functions you
should not have to worry. The names of the array and coordinates is also
consistent with that of :external+xhistogram:doc:`xhistogram <index>`. Only the
right edge attribute will be missing.

.. admonition:: Right edge inference

   If the right edge attribute is missing in a bins coordinates, the accessor
   will try to infer it. It will make the hypothesis that bins are regularly
   spaced. If this is not the case, an exception will be raised.
