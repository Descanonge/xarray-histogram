.. currentmodule:: xarray_histogram

********
Accessor
********

An :external+xarray:doc:`accessor <internals/extending-xarray>` is provided to
ease manipulation and analysis of the histogram outputs. Simply import
:mod:`xarray_histogram.accessor` to register it. It will then be available for
all DataArrays that meet some conditions (see below), under the ``hist``
attribute. It gives access to a number of methods. ::

    import xarray_histogram as xh
    import xarray_histogram.accessor

    h = xh.histogram(data)

    h.hist.median()

Operations are vectorized [#vector]_, so that you can apply them to entire
arrays of histograms. For instance for data defined along time, latitude and
longitude, we can compute one histogram per time-step::

    >>> h = xh.histogram(data, dims=["lon", "lat"])
    >>> h.hist.median()
    will be of dimensions ("time",)

.. [#vector] Computations are automatically vectorized in Python with
   :func:`xarray.apply_ufunc`, which is not efficient for a large number of
   histograms.


Conditions of accessibility
===========================

Once registered, an accessor is a cached property that can be accessed on any
DataArray. They are some conditions for the *hist* accessor to be created
successfully:

* The coordinates of the bins must be named ``<variable>_bins``.
* The array must be named as ``<variable(s)_name>_<histogram or pdf>``.
  *histogram* if it is not normalized, and *pdf* if it is normalized as a
  probability density function. If the histogram is multi-dimensional, the
  variables names must be separated by underscores. For instance:
  ``Temp_Sal_histogram``.

Each bins coordinate may contain attributes:

* ``bin_type``: the class name of the Boost axis type that was used. If not
  present, the accessor will assume the bins are regularly spaced and will try
  to infer the rightmost edge.
* ``right_edge``: the rightmost edge position, only necessary for Regular and
  Variable bins.
* ``underflow`` and ``overflow``: booleans that indicate if the corresponding
  flow bins are present. If not present, will assume no flow bins.

Those conventions are coherent with the output of
``xarray_histogram.histogram*``, so if you use this package functions you
should not have to worry. The names of the array and coordinates is also
consistent with that of :external+xhistogram:doc:`xhistogram <index>`
(although coordinates attributes will be missing).

Computations
============

Bins
----

The accessor provides a number of methods that return bins-related values for a
given variable. If the histogram is uni-dimensional (*ie* for a single variable)
the variable name can be omitted. By default flow bins are kept but they can be
excluded by passing ``flow=False``.

* :meth:`~.HistDataArrayAccessor.bins` returns the corresponding coordinate,
  this is essentially ``h.hist.coords["var_bins"]``.

* :meth:`~.HistDataArrayAccessor.edges` returns the N+1 edges (including the
  rightmost edge). Edges are not available for the discrete bins "IntCategory"
  and "StrCategory".

* :meth:`~.HistDataArrayAccessor.widths` returns the widths of the bins
  The widths of flow bins and StrCategory are always 1.

* :meth:`~.HistDataArrayAccessor.centers` returns the center position of the
  bins. The overflow bins centers are the same as their position (``np.inf`` for
  instance).

* :meth:`~.HistDataArrayAccessor.areas` returns the areas of multidimensional
  bins. This is the product of the widths of all bins. Only some variable can be
  specified. The areas of points that correspond to a flow bin in at least one
  dimension is equal to one. For instance for a 2D-histogram with underflow and
  overflow bins, all the borders of the 2D array for areas will be equal to 1.

To remove flow bins, :meth:`~.HistDataArrayAccessor.remove_flow` will returns a
new histogram DataArray without the flow bins of the given variables (by default
all of them). This simply does a ``.isel`` operation based on the ``underflow``
and ``overflow`` attributes of specified coordinates. It also set those
attributes to False in the output.

Bins transform
--------------

Arbitrary functions can be applied to bins with
:meth:`~.HistDataArrayAccessor.apply_func`. The result is equivalent to
computing an histogram of ``func(data["variable"])``. The function must
transform the N+1 edges given as a DataArray. There is no need to account for
the *right_edge* attribute.

For instance, :meth:`~.HistDataArrayAccessor.scale` scales bins by a given
factor. It essential does ``hist.apply_func(lambda edges: edges * factor)``


Normalization
-------------

The histogram can be normalized to a probability density function if not
already, using :meth:`~.HistDataArrayAccessor.normalize`. Note that for a
N-dimensional histogram, this function can normalize only along some variables.

The accessor considers the histogram normalized or not given the name of its
DataArray: normalized if named ``<variables>_pdf`` and non-normalized
if ``<variables>_histogram``. This is consistent with the output of
:func:`~.core.histogram`.

.. important::

   This is important when computing statistics (see below) where the accessor
   must know if the histogram is normalized or not.

Normalizing when flow bins are present in the output is allowed. The values in
flow bins are not changed and not counted in the normalization.

Statistics
----------

A number of statistics can be extracted from the histogram. The following
functions are wrappers around methods of :class:`scipy.stats.rv_histogram`.
These function work only on 1D histograms, thus for ND-histograms a variable
must be specified. This does not support flow bins, they are removed along the
core dimension (the specified variable).

.. note::

    The histogram cannot be chunked in the core dimension.

.. autosummary::

   ~accessor.HistDataArrayAccessor.cdf
   ~accessor.HistDataArrayAccessor.interval
   ~accessor.HistDataArrayAccessor.median
   ~accessor.HistDataArrayAccessor.mean
   ~accessor.HistDataArrayAccessor.moment
   ~accessor.HistDataArrayAccessor.ppf
   ~accessor.HistDataArrayAccessor.std
   ~accessor.HistDataArrayAccessor.var


