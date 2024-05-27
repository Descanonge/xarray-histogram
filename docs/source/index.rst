
XArray-Histogram documentation
==============================

This package allow to compute histograms from XArray data, taking advantage of its label and dimensions management.
It relies on the `Boost Histogram <https://boost-histogram.readthedocs.io>`_ library for the computations.

It is essentially a thin wrapper using directly Boost Histogram on loaded data, or `Dask-histogram <https://dask-histogram.readthedocs.io>`__ on data contained in dask arrays. It thus features optimised performance, as well as lazy computation and easy up-scaling thanks to dask.


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

   usage

.. toctree::
   :maxdepth: 1

   api

Source code: `<https://github.com/Descanonge/xarray-histogram>`__

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
