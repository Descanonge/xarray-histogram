
XArray-Histogram documentation
==============================

This package allow to compute histograms from XArray data, taking advantage of its label and dimensions management.
It relies on the `Boost Histogram <https://boost-histogram.readthedocs.io>`_ library for the computations.

It is essentially a thin wrapper using directly Boost Histogram on loaded data, or `Dask-histogram <https://dask-histogram.readthedocs.io>`__ on data contained in dask arrays. It thus features optimised performance, as well as lazy computation and easy up-scaling thanks to dask.

Quick examples
--------------

Bins can be specified similarly to numpy functions::

   import xarray_histogram as xh
   hist = xh.histogram(data, bins=(100, 0., 10.))

but also using boost `axes <https://boost-histogram.readthedocs.io/en/latest/user-guide/axes.html>`__, benefiting from their features::

   import boost_histogram as bh
   hist = xh.histogram(data, bins=[bh.axis.Regular(100, 0., 10.)])

Multi-dimensional histogram can be computed, here in 2D for instance::

   hist = xh.histogram(
      temp, sal,
      bins=[bh.axis.Regular(100, -5., 40.), bh.axis.Regular(100, 31, 38))
   )

Finally, so far we have computed histograms on the whole flattened arrays, but we can compute only along some dimensions. For instance we can retrieve the time evolution of an histogram::

   hist = xh.histogram(temp, bins=[bh.axis.Regular(100, 0., 10.)], dims=['lat', 'lon'])

Histograms can be normalized, and weights can be applied.
All of this works seamlessly with data stored in numpy or dask arrays.

Installation
------------

From PyPi: at some point...🚧

From source::

  git clone https://github.com/Descanonge/xarray-histogram
  cd xarray-histogram
  pypi install -e .

or::

 pip install https://github.com/Descanonge/xarray-histogram.git#egg=xarray-histogram


Requirements
------------

- Python >= 3.7
- numpy
- xarray
- `boost-histogram <https://github.com/scikit-hep/boost-histogram>`__

Optional dependencies
---------------------

If not available all data will be eagerly loaded.

- `dask <https://www.dask.org/>`__
- `dask-histogram <https://github.com/dask-contrib/dask-histogram>`__

.. toctree::
   :maxdepth: 1

   api

Source code: `<https://github.com/Descanonge/xarray-histogram>`__

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
